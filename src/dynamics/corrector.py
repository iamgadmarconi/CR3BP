import numba
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

from .propagator import crtbp_accel, propagate_crtbp


import numpy as np

def halo_diff_correct(x0_guess, mu, case=2, tol=1e-12, max_iter=25):
    """
    Replicate haloget(..., CASE=2) from MATLAB "exactly".
    That means:
      - We fix x0 (the initial x-position).
      - We iterate on z0, vy0.
      - We want y=0 crossing with vx=0, vz=0, i.e. Dx1=0, Dz1=0.
      - We also incorporate the 'DDx1, DDz1' terms in the Newton iteration.
    """
    # Create a copy of the initial guess to avoid modifying the input
    x0 = np.copy(x0_guess)
    
    # If we're getting extreme values, the initial state might need normalization
    # Make sure x0 has the format [x, y, z, vx, vy, vz] with y=0 and vx=vz=0
    if len(x0) != 6:
        raise ValueError("x0 must be a 6-element array [x,y,z,vx,vy,vz]")
    
    # Ensure y=0 and vx=vz=0 for a proper halo initial condition
    x0[1] = 0.0  # y
    x0[3] = 0.0  # vx
    x0[5] = 0.0  # vz
    
    iteration = 0
    mu1 = 1.0 - mu
    mu2 = mu
    
    while True:
        iteration += 1
        if iteration > max_iter:
            raise RuntimeError(f"halo_diff_correct: did not converge after {max_iter} iterations")
    
        # 1) Find next y=0 crossing using the new method
        t_cross, X_cross = find_y_crossing(x0, mu, guess_t=np.pi/2, tol=tol)
        
        # Print debug info
        # print(f"Iteration {iteration}:")
        # print(f"  t_cross = {t_cross}")
        # print(f"  X_cross = {X_cross}")
        
        # Unpack final crossing states
        x1, y1, z1, vx1, vy1, vz1 = X_cross
        
        # Evaluate the difference (the "errors") we want to drive to zero
        Dx1 = vx1
        Dz1 = vz1
        
        # Check if we've converged within tolerance
        if abs(Dx1) < tol and abs(Dz1) < tol:
            return x0, t_cross
        
        # 2) Compute the necessary derivative terms for the correction
        r1 = np.sqrt((x1+mu)**2 + y1**2 + z1**2)
        r2 = np.sqrt((x1 - (1.0 - mu))**2 + y1**2 + z1**2)
        rho1 = 1.0 / (r1**3)
        rho2 = 1.0 / (r2**3)
        
        # Calculate omgx1 (partial derivative of potential wrt x)
        omgx1 = - (mu1*(x1+mu)*rho1) - (mu2*(x1 - (1.0 - mu))*rho2) + x1
        
        # Calculate DDx1 (acceleration in x direction)
        DDx1 = 2.0*vy1 + omgx1
        
        # Calculate DDz1 (acceleration in z direction)
        DDz1 = - (mu1*z1*rho1) - (mu2*z1*rho2)
        
        # 3) Compute the STM from t=0 to t=t_cross
        _, _, Phi_final, _ = compute_stm(x0, mu, t_cross, atol=tol, rtol=tol)
        # Extract relevant partial derivatives from Phi_final
        phi_vx_z0  = Phi_final[3, 2]  # phi(4,3) in MATLAB indexing
        phi_vx_vy0 = Phi_final[3, 4]  # phi(4,5)
        phi_vz_z0  = Phi_final[5, 2]  # phi(6,3)
        phi_vz_vy0 = Phi_final[5, 4]  # phi(6,5)
        phi_y_z0   = Phi_final[1, 2]  # phi(2,3)
        phi_y_vy0  = Phi_final[1, 4]  # phi(2,5)

        # 4) Build the correction matrix as in MATLAB
        C1 = np.array([[phi_vx_z0, phi_vx_vy0],
                        [phi_vz_z0, phi_vz_vy0]])
        
        # Construct the outer product term
        dd_vec = np.array([[DDx1], [DDz1]])  # Column vector
        phi_vec = np.array([[phi_y_z0, phi_y_vy0]])  # Row vector
        outer_product = dd_vec @ phi_vec  # Explicitly calculate the outer product
        
        # Ensure vy1 is not too close to zero
        if abs(vy1) < 1e-14:
            raise RuntimeError("Cannot do the (1/Dy1) correction if vy1 is near zero.")
        
        # Calculate the corrected matrix C2
        C2 = C1 - (1.0/vy1) * outer_product
        
        # 5) Solve the 2×2 linear system for the corrections
        RHS = np.array([-Dx1, -Dz1])
        dU = np.linalg.solve(C2, RHS)
        dz0  = dU[0]
        dvy0 = dU[1]
        
        # 6) Update x0 in place (only z0 and vy0 for CASE=2)
        x0[2] += dz0   # z0
        x0[4] += dvy0  # vy0
        
        # Then we loop back until abs(Dx1) and abs(Dz1) are < tol

def lyapunov_diff_correct(x0_guess, mu, tol=1e-12, max_iter=50):
    """
    Differential corrector for a planar Lyapunov-like orbit.
    Keeps x0, y0, z0, vx0, vz0 fixed, adjusting only vy0
    so that the next crossing of y=0 has vx(t_cross) = 0 (or some condition).
    
    Returns (x0_corrected, half_period).
    """
    x0 = np.copy(x0_guess)
    attempt = 0
    
    while True:
        attempt += 1
        if attempt > max_iter:
            raise RuntimeError("Max attempts exceeded in differential corrector.")
        
        # 1) Find next y=0 crossing from some guess time. 
        #    We can guess half a period near pi, or adapt as you see fit
        guess_t = np.pi
        t_cross, X_cross = find_y_crossing(x0, mu, guess_t, direction=1)
        
        # The crossing states
        x_cross = X_cross[0]
        y_cross = X_cross[1]
        z_cross = X_cross[2]
        vx_cross = X_cross[3]
        vy_cross = X_cross[4]
        vz_cross = X_cross[5]
        
        # Condition we want: e.g., vx_cross = 0 at y=0 crossing
        # If it's close to zero, we stop
        if abs(vx_cross) < tol:
            # Done
            half_period = t_cross
            return x0, half_period
        
        # Otherwise, we do a Newton step:
        # Evaluate the STM at t_cross to see how vx(t_cross) depends on vy0
        # We'll do the same integration but with the STM.
        PHI0 = np.zeros(42)
        PHI0[:6] = x0
        PHI0[6:] = np.eye(6).flatten()
        
        def ode_fun(t, y):
            return variational_equations(t, y, mu)
        
        # We'll integrate to the crossing time directly
        sol = solve_ivp(ode_fun, [0, t_cross], PHI0, rtol=1e-12, atol=1e-12, dense_output=True)
        
        # Evaluate the final 42-vector at t_cross
        PHI_vec_final = sol.sol(t_cross)
        # The final state:
        state_final = PHI_vec_final[:6]
        # Flattened STM
        phi_final = PHI_vec_final[6:].reshape((6,6))
        
        # We want partial of vx_cross wrt initial vy. That is phi_final(3,4) in 0-based indexing:
        # ( Row=3 for vx, Col=4 for vy ), if all other initial states are held fixed.
        dvx_dvy0 = phi_final[3,4]
        
        # Basic linear correction:  dvy = - vx_cross / dvx_dvy0
        dvy = - vx_cross / dvx_dvy0
        
        # Update x0
        x0[4] += dvy  # adjust the initial vy


def halo_y(t1, x0, mu):
    """
    Returns the y-position of the halo orbit at time t1, starting from initial state x0 at t=0.
    """
    sol = propagate_crtbp(x0, 0.0, t1, mu)
    states = sol.y.T
    state_t1 = states[-1]
    y_position = state_t1[1]
    return np.array(y_position, dtype=np.float64)  # return as float64 array/scalar

@numba.njit(fastmath=True, cache=True)
def jacobian_crtbp(x, y, z, mu):
    """
    Returns the 6x6 Jacobian matrix F for the 3D CRTBP in the rotating frame,
    mirroring the MATLAB var3D.m approach (with mu2 = 1 - mu).
    
    The matrix F is structured as:
         [ 0    0    0    1     0    0 ]
         [ 0    0    0    0     1    0 ]
         [ 0    0    0    0     0    1 ]
         [ omgxx omgxy omgxz  0     2    0 ]
         [ omgxy omgyy omgyz -2     0    0 ]
         [ omgxz omgyz omgzz  0     0    0 ]

    Indices: x=0, y=1, z=2, vx=3, vy=4, vz=5

    This matches the partial derivatives from var3D.m exactly.
    """

    # As in var3D.m:
    #   mu2 = 1 - mu (big mass fraction)
    mu2 = 1.0 - mu

    # Distances squared to the two primaries
    # r^2 = (x+mu)^2 + y^2 + z^2       (distance^2 to M1, which is at (-mu, 0, 0))
    # R^2 = (x - mu2)^2 + y^2 + z^2    (distance^2 to M2, which is at (1-mu, 0, 0))
    r2 = (x + mu)**2 + y**2 + z**2
    R2 = (x - mu2)**2 + y**2 + z**2
    r3 = r2**1.5
    r5 = r2**2.5
    R3 = R2**1.5
    R5 = R2**2.5

    # From var3D.m, the partial derivatives "omgxx," "omgyy," ...
    omgxx = 1.0 \
        + mu2/r5 * 3.0*(x + mu)**2 \
        + mu  /R5 * 3.0*(x - mu2)**2 \
        - (mu2/r3 + mu/R3)

    omgyy = 1.0 \
        + mu2/r5 * 3.0*(y**2) \
        + mu  /R5 * 3.0*(y**2) \
        - (mu2/r3 + mu/R3)

    omgzz = 0.0 \
        + mu2/r5 * 3.0*(z**2) \
        + mu  /R5 * 3.0*(z**2) \
        - (mu2/r3 + mu/R3)

    omgxy = 3.0*y * ( mu2*(x + mu)/r5 + mu*(x - mu2)/R5 )
    omgxz = 3.0*z * ( mu2*(x + mu)/r5 + mu*(x - mu2)/R5 )
    omgyz = 3.0*y*z*( mu2/r5 + mu/R5 )

    # Build the 6x6 matrix F
    F = np.zeros((6, 6), dtype=np.float64)

    # Identity block for velocity wrt position
    F[0, 3] = 1.0  # dx/dvx
    F[1, 4] = 1.0  # dy/dvy
    F[2, 5] = 1.0  # dz/dvz

    # The second derivatives block
    F[3, 0] = omgxx
    F[3, 1] = omgxy
    F[3, 2] = omgxz

    F[4, 0] = omgxy
    F[4, 1] = omgyy
    F[4, 2] = omgyz

    F[5, 0] = omgxz
    F[5, 1] = omgyz
    F[5, 2] = omgzz

    # Coriolis terms
    F[3, 4] = 2.0
    F[4, 3] = -2.0

    return F

@numba.njit(fastmath=True, cache=True)
def variational_equations(t, PHI_vec, mu, forward=1):
    """
    3D variational equations for the CR3BP, matching MATLAB's var3D.m layout.
    
    PHI_vec is a 42-element vector:
      - PHI_vec[:36]   = the flattened 6x6 STM (Phi)
      - PHI_vec[36:42] = the state vector [x, y, z, vx, vy, vz]
    
    We compute d/dt(PHI_vec).
    The resulting derivative is also 42 elements:
      - first 36 = dPhi/dt (flattened)
      - last 6 = [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt] * forward

    This function calls 'jacobian_crtbp_matlab(...)' to build the 6x6 matrix F
    and then does the same matrix multiplication as in var3D.m:

        phidot = F * Phi
    """
    # 1) Unpack the STM (first 36) and the state (last 6)
    phi_flat = PHI_vec[:36]
    x_vec    = PHI_vec[36:]  # [x, y, z, vx, vy, vz]

    # Reshape the STM to 6x6
    Phi = phi_flat.reshape((6, 6))

    # Unpack the state
    x, y, z, vx, vy, vz = x_vec

    # 2) Build the 6x6 matrix F from the partial derivatives
    F = jacobian_crtbp(x, y, z, mu)

    # 3) dPhi/dt = F * Phi  (manually done to keep numba happy)
    phidot = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            s = 0.0
            for k in range(6):
                s += F[i, k] * Phi[k, j]
            phidot[i, j] = s

    # 4) State derivatives, same formula as var3D.m
    #    xdot(4) = x(1) - mu2*( (x+mu)/r3 ) - mu*( (x-mu2)/R3 ) + 2*vy, etc.
    mu2 = 1.0 - mu
    r2 = (x + mu)**2 + y**2 + z**2
    R2 = (x - mu2)**2 + y**2 + z**2
    r3 = r2**1.5
    R3 = R2**1.5

    ax = ( x 
           - mu2*( (x+mu)/r3 ) 
           -  mu*( (x - mu2)/R3 ) 
           + 2.0*vy )
    ay = ( y
           - mu2*( y / r3 )
           -  mu*( y / R3 )
           - 2.0*vx )
    az = ( - mu2*( z / r3 ) 
           - mu  *( z / R3 ) )

    # 5) Build derivative of the 42-vector
    dPHI_vec = np.zeros_like(PHI_vec)

    # First 36 = flattened phidot
    dPHI_vec[:36] = phidot.ravel()

    # Last 6 = [vx, vy, vz, ax, ay, az], each multiplied by 'forward'
    dPHI_vec[36] = forward * vx
    dPHI_vec[37] = forward * vy
    dPHI_vec[38] = forward * vz
    dPHI_vec[39] = forward * ax
    dPHI_vec[40] = forward * ay
    dPHI_vec[41] = forward * az

    return dPHI_vec

def integrate_variational_equations(PHI_init, mu, T, forward=1, 
                                  rtol=1e-12, atol=1e-12, steps=1000):
    """
    Integrate the 42D system from t=0 to t=T (a positive T).
    The 'forward' parameter (+1 or -1) controls direction of the state derivatives.
    If forward=-1, we can optionally flip the sign of the output times to reflect a negative timescale.
    
    PHI_init: 42-vector with MATLAB-like layout:
        PHI_init[:36]  = flattened 6x6 STM
        PHI_init[36:42] = [x, y, z, vx, vy, vz]
    mu: mass ratio
    T:  final time (must be > 0 for solve_ivp's standard approach)
    forward: +1 or -1
    """
    def rhs(t, PHI):
        return variational_equations(t, PHI, mu, forward)

    t_span = (0.0, T)
    t_eval = np.linspace(0.0, T, steps)

    sol = solve_ivp(rhs, t_span, PHI_init, t_eval=t_eval, 
                    rtol=rtol, atol=atol, vectorized=False)

    # If you want the time array to reflect backward integration for forward=-1:
    if forward == -1:
        sol.t = -sol.t  # times run 0 -> -T, matching MATLAB's "t=FORWARD*t" logic

    return sol


def compute_stm(x0, mu, tf, forward=1, **solve_kwargs):
    """
    Integrate the 3D CRTBP + STM from t=0 to t=tf, mirroring MATLAB's var3D layout.
    
    The integrated vector is 42 elements:
      - first 36 = flattened 6x6 identity matrix (initial STM),
      - last 6   = [x, y, z, vx, vy, vz].

    Arguments:
    ----------
    x0        : 6-element array of initial conditions (3D CRTBP)
    mu        : mass ratio
    tf        : final integration time (positive)
    forward   : +1 (forward in time) or -1 (reverse integration)
    solve_kwargs : additional keyword arguments for solve_ivp (e.g. rtol, atol)

    Returns:
    --------
    x         : (n_times x 6) array of the integrated state over [0, tf]
    t         : (n_times,) array of times (flipped to negative if forward=-1)
    phi_T     : 6x6 monodromy matrix at t=tf (flattened portion of last row)
    PHI       : (n_times x 42) full integrated solution, where each row is
                [flattened STM(36), state(6)] in that order.
    """

    # Build initial 42-vector in MATLAB ordering: [flattened STM, state]
    PHI0 = np.zeros(42, dtype=np.float64)
    # The first 36 = identity matrix
    PHI0[:36] = np.eye(6, dtype=np.float64).ravel()
    # The last 6 = x0
    PHI0[36:] = x0

    # Set default solver tolerances if not provided
    if 'rtol' not in solve_kwargs:
        solve_kwargs['rtol'] = 3e-14
    if 'atol' not in solve_kwargs:
        solve_kwargs['atol'] = 1e-14

    def ode_fun(t, y):
        # Calls our Numba-accelerated function
        return variational_equations(t, y, mu, forward)

    # Integrate from 0 to tf
    t_span = (0.0, tf)
    sol = solve_ivp(ode_fun, t_span, PHI0, **solve_kwargs)

    # Possibly flip time if forward==-1, to mirror MATLAB's t=FORWARD*t
    if forward == -1:
        sol.t = -sol.t  # so we see times from 0 down to -tf

    # Reformat outputs
    t = sol.t
    PHI = sol.y.T      # shape (n_times, 42)
    
    # The state is in columns [36..41] of PHI
    x = PHI[:, 36:42]   # shape (n_times, 6)

    # The final row's first 36 columns = flattened STM at t=tf
    phi_tf_flat = PHI[-1, :36]
    phi_T = phi_tf_flat.reshape((6, 6))

    return x, t, phi_T, PHI
    return x, t, phi_T, PHI