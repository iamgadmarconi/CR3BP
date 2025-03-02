import numba
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

from .propagator import crtbp_accel


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


def find_y_crossing(x0, mu, guess_t, direction=None, tol=1e-12, max_steps=1000):
    """
    Find y=0 crossing using an approach similar to MATLAB's haloy/find0.
    Uses a root-finding approach to precisely locate the crossing time.
    """
    # Function to get y-position at any time t
    def get_y_at_time(t):
        if t <= 0:  # Prevent looking at t=0 or negative time
            return x0[1]
        
        # Integrate from 0 to t
        sol = solve_ivp(lambda t, y: crtbp_accel(y, mu),
                       [0, t], x0,
                       rtol=tol, atol=tol,
                       dense_output=True,
                       method='RK45')
        
        # Get final state
        y = sol.sol(t)[1]  # y-component
        return y
    
    # First, let's sample some points to find where y changes sign
    t_samples = np.linspace(0.1, 2*np.pi, 50)
    y_samples = [get_y_at_time(t) for t in t_samples]
    
    # Find where y changes sign
    for i in range(len(t_samples)-1):
        if y_samples[i] * y_samples[i+1] <= 0:  # Sign change detected
            t_start = t_samples[i]
            t_end = t_samples[i+1]
            
            # Use root_scalar to find the precise crossing time
            try:
                result = root_scalar(get_y_at_time,
                                   bracket=[t_start, t_end],
                                   method='brentq',
                                   rtol=tol)
                
                if not result.converged:
                    continue  # Try next interval if this one fails
                
                t_cross = result.root
                
                # Get the full state at the crossing
                sol = solve_ivp(lambda t, y: crtbp_accel(y, mu),
                              [0, t_cross], x0,
                              rtol=tol, atol=tol,
                              dense_output=True,
                              method='RK45')
                
                X_cross = sol.sol(t_cross)
                
                # If we found a crossing near t=0, keep looking
                if t_cross < 0.1:
                    continue
                    
                return t_cross, X_cross
                
            except ValueError:
                continue  # Try next interval if bracketing fails
    
    raise RuntimeError("Could not find any y=0 crossing")

@numba.njit(fastmath=True, cache=True)
def jacobian_crtbp(x, y, z, mu):
    """
    Returns the 6x6 Jacobian of the CR3BP vector field
    f(X) = [vx, vy, vz, ax, ay, az]
    with respect to X = [x, y, z, vx, vy, vz].
    
    This includes the partials of (ax, ay, az) wrt (x,y,z)
    PLUS the standard rotating-frame terms that couple velocity.
    """
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    
    # To avoid repeated calls:
    r1_3 = r1**3
    r1_5 = r1**5
    r2_3 = r2**3
    r2_5 = r2**5
    
    # Derivatives of the CR3BP acceleration in rotating frame:
    # ax = 2*vy + x - (1 - mu)*(x + mu)/r1^3 - mu*(x - 1 + mu)/r2^3
    # We want partial derivatives wrt x,y,z of ax, ay, az.
    
    # Common factors:
    mu1 = 1.0 - mu
    mu2 = mu
    
    # partial wrt x
    dUxx = (1
            - mu1*(1.0/r1_3 - 3.0*(x+mu)**2 / r1_5)
            - mu2*(1.0/r2_3 - 3.0*(x - 1 + mu)**2 / r2_5))
    dUxy = (0
            + 3.0*mu1*(x+mu)*y / r1_5
            + 3.0*mu2*(x-1+mu)*y / r2_5)
    dUxz = (0
            + 3.0*mu1*(x+mu)*z / r1_5
            + 3.0*mu2*(x-1+mu)*z / r2_5)
    
    # partial wrt y
    dUyx = dUxy  # symmetry for second partial derivatives
    dUyy = (1
            - mu1*(1.0/r1_3 - 3.0*y**2 / r1_5)
            - mu2*(1.0/r2_3 - 3.0*y**2 / r2_5))
    dUyz = (0
            + 3.0*mu1*y*z / r1_5
            + 3.0*mu2*y*z / r2_5)
    
    # partial wrt z
    dUzx = dUxz
    dUzy = dUyz
    dUzz = (0
            - mu1*(1.0/r1_3 - 3.0*z**2 / r1_5)
            - mu2*(1.0/r2_3 - 3.0*z**2 / r2_5))
    
    # Construct the 6x6 Jacobian matrix:
    jac = np.zeros((6, 6), dtype=np.float64)
    
    # Position derivatives
    jac[0, 3] = 1.0
    jac[1, 4] = 1.0
    jac[2, 5] = 1.0
    
    # Acceleration derivatives with respect to position
    jac[3, 0] = dUxx
    jac[3, 1] = dUxy
    jac[3, 2] = dUxz
    jac[4, 0] = dUyx
    jac[4, 1] = dUyy
    jac[4, 2] = dUyz
    jac[5, 0] = dUzx
    jac[5, 1] = dUzy
    jac[5, 2] = dUzz
    
    # Coriolis terms
    jac[3, 4] = 2.0
    jac[4, 3] = -2.0
    
    return jac

@numba.njit(fastmath=True, cache=True)
def variational_equations(t, PHI_vec, mu):
    """
    3D variational equations for the CR3BP.
    
    PHI_vec is a 42-element vector:
       first 6 = [x, y, z, vx, vy, vz]
       next 36 = the flattened 6x6 STM, Phi(t).
    
    Returns d/dt(PHI_vec).
    """
    # Unpack the state and the STM
    x = PHI_vec[0]
    y = PHI_vec[1]
    z = PHI_vec[2]
    vx = PHI_vec[3]
    vy = PHI_vec[4]
    vz = PHI_vec[5]
    
    # Create state array for crtbp_accel
    state = np.array([x, y, z, vx, vy, vz], dtype=np.float64)
    
    # The 36 entries of the STM
    phi_flat = PHI_vec[6:]
    phi = phi_flat.reshape((6, 6))
    
    # Compute the 6D derivative of the state
    state_dot = crtbp_accel(state, mu)
    
    # Compute the 6x6 Jacobian wrt x,y,z,vx,vy,vz
    dfmat = jacobian_crtbp(x, y, z, mu)
    
    # Variational equation: dPhi/dt = dfmat * Phi
    phi_dot = np.zeros((6, 6), dtype=np.float64)
    # Manual matrix multiplication for better Numba compatibility
    for i in range(6):
        for j in range(6):
            for k in range(6):
                phi_dot[i, j] += dfmat[i, k] * phi[k, j]
    
    # Flatten phi_dot and put it all together
    dPHI_vec = np.zeros_like(PHI_vec)
    # The first 6 are the time derivatives of the state
    dPHI_vec[:6] = state_dot
    # The next 36 are the flatten of phi_dot
    dPHI_vec[6:] = phi_dot.ravel()
    return dPHI_vec

def compute_stm(x0, mu, tf, **solve_kwargs):
    """
    Integrate the 3D CR3BP plus STM from t=0 to t=tf,
    returning (x, t, phi_T, PHI) where:
      x      : state trajectory (each row is [x,y,z,vx,vy,vz])
      t      : time array corresponding to x
      phi_T  : final 6x6 state transition (monodromy) matrix at t=tf
      PHI    : the full integrated solution, where each row is 
               [state, flattened STM]
    
    x0 is a 6-vector of initial conditions.
    The ordering in the integrated vector remains as [state, flattened STM].
    """
    # Build initial 42-vector: [x0, reshape(I_6)]
    PHI0 = np.zeros(6 + 36)
    PHI0[:6] = x0
    PHI0[6:] = np.eye(6).flatten()
    
    def ode_fun(t, y):
        return variational_equations(t, y, mu)
    
    # Set default tolerance values if not provided
    if 'rtol' not in solve_kwargs:
        solve_kwargs['rtol'] = 3e-14
    if 'atol' not in solve_kwargs:
        solve_kwargs['atol'] = 1e-14
    
    sol = solve_ivp(ode_fun, [0, tf], PHI0, **solve_kwargs)
    
    # The entire trajectory + STM (each row: [state, flattened STM])
    t_array = sol.t
    PHI = sol.y.T  # shape (len(t_array), 42)
    
    # Extract the state trajectory (first 6 columns)
    x = PHI[:, :6]
    
    # Extract the final state transition matrix (monodromy matrix)
    phi_tf_flat = PHI[-1, 6:]
    phi_T = phi_tf_flat.reshape((6, 6))
    
    return x, t_array, phi_T, PHI
