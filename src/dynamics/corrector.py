import numba
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar, fsolve

from .propagator import crtbp_accel, propagate_crtbp



def lyapunov_diff_correct(x0_guess, mu, forward=1, tol=1e-12, max_iter=250):
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

        t_cross, X_cross = find_x_crossing(x0, mu, forward=forward)
        
        # The crossing states
        x_cross = X_cross[0]
        y_cross = X_cross[1]
        z_cross = X_cross[2]
        vx_cross = X_cross[3]
        vy_cross = X_cross[4]
        vz_cross = X_cross[5]
        
        if abs(vx_cross) < tol:
            # Done
            half_period = t_cross
            return x0, half_period
        
        # Build the initial condition vector for the combined state and STM.
        # New convention:
        #   First 36 elements: flattened 6x6 STM (initialized to identity)
        #   Last 6 elements: the state vector x0
        PHI0 = np.zeros(42)
        PHI0[:36] = np.eye(6).flatten()
        PHI0[36:] = x0
        
        def ode_fun(t, y):
            return variational_equations(t, y, mu, forward=forward)
        
        # We'll integrate to the crossing time directly
        sol = solve_ivp(ode_fun, [0, t_cross], PHI0, rtol=1e-12, atol=1e-12, dense_output=True)
        
        # Evaluate the final 42-vector at t_cross
        PHI_vec_final = sol.sol(t_cross)
        
        # Extract the STM and the final state using the new convention.
        phi_final = PHI_vec_final[:36].reshape((6, 6))
        state_final = PHI_vec_final[36:]
        
        # We want the partial derivative of vx_cross with respect to the initial vy.
        # This is given by phi_final[3,4] in 0-based indexing.
        dvx_dvy0 = phi_final[3, 4]
        
        # Basic linear correction:  dvy = - vx_cross / dvx_dvy0
        dvy = -vx_cross / dvx_dvy0
        
        # Update x0 (adjust only the vy component)
        x0[4] += dvy

def _find_bracket(f, x0, max_expand=500):
    """
    Attempt to find a bracket around x0 in a manner similar to MATLAB's fzero.
    It starts with an initial step of 2% of |x0| (or 0.02 if x0==0) and expands
    symmetrically in both directions (multiplying the step by sqrt(2) each iteration)
    until it finds an interval where f changes sign.

    Parameters
    ----------
    f : callable
        The function for which we want a root f(x)=0.
    x0 : float
        Initial guess.
    max_expand : int
        Maximum number of expansions to try.

    Returns
    -------
    root : float
        A solution of f(root)=0 found using Brent's method on the bracket.
    """
    f0 = f(x0)
    if abs(f0) < 1e-14:
        return x0

    dx = 1e-10 # * abs(x0) if x0 != 0 else 1e-10

    for _ in range(max_expand):
        # Try the positive direction: x0 + dx
        x_right = x0 + dx
        f_right = f(x_right)
        if np.sign(f_right) != np.sign(f0):
            a, b = (x0, x_right) if x0 < x_right else (x_right, x0)
            return root_scalar(f, bracket=(a, b), method='brentq', xtol=1e-12).root

        # Try the negative direction: x0 - dx
        x_left = x0 - dx
        f_left = f(x_left)
        if np.sign(f_left) != np.sign(f0):
            a, b = (x_left, x0) if x_left < x0 else (x0, x_left)
            return root_scalar(f, bracket=(a, b), method='brentq', xtol=1e-12).root

        # Expand step size by multiplying by sqrt(2)
        dx *= np.sqrt(2)

def find_x_crossing(x0, mu, forward=1):
    """
    From the given state x0, find the next time t1_z at which the orbit crosses y=0.
    Returns the crossing time (t1_z) and the corresponding state (x1_z).
    This replaces the global-variable version of find0.m.
    """
    tolzero = 1.e-10
    # Initial guess for the time
    t0_z = np.pi/2 - 0.15

    # 1) Integrate from t=0 up to t0_z.
    sol = propagate_crtbp(x0, 0.0, t0_z, mu, forward=forward, steps=500)
    xx = sol.y.T  # assume sol.y is (state_dim, time_points)
    x0_z = xx[-1]  # final state after integration

    # 2) Define a local function that depends on time t.
    def halo_y_wrapper(t):
        return halo_y(t, t0_z, x0_z, mu, forward=forward, steps=500)

    # 3) Find the time at which y=0 by bracketing the root.
    t1_z = _find_bracket(halo_y_wrapper, t0_z)

    # 4) Integrate from t0_z to t1_z to get the final state.
    sol = propagate_crtbp(x0_z, t0_z, t1_z, mu, forward=forward, steps=500)
    xx_final = sol.y.T
    x1_z = xx_final[-1]

    return t1_z, x1_z


def halo_y(t1, t0_z, x0_z, mu, forward=1, steps=3000, tol=1e-10):
    """
    Python equivalent of the MATLAB haloy() function.
    
    Parameters
    ----------
    t1    : float
            Time at which we want the y-value of the halo orbit
    t0_z  : float
            Reference "start time"
    x0_z  : array-like, shape (6,)
            Reference "start state" for the halo
    mu    : float
            CRTBP mass parameter
    forward : +1 or -1
            Direction of integration
    steps : int
            How many points in the time array
    tol   : float
            Tolerance for the ODE solver

    Returns
    -------
    y1 : float
         The y-component of the halo orbit state at t1
    """
    # If t1 == t0_z, no integration is done.  Just take the initial condition.
    if np.isclose(t1, t0_z, rtol=3e-10, atol=1e-10):
        x1_zgl = x0_z
    else:
        sol = propagate_crtbp(x0_z, t0_z, t1, mu, forward=forward, steps=steps, rtol=3*tol, atol=tol)
        xx = sol.y.T
        # The final state is the last row of xx
        x1_zgl = xx[-1, :]

    # x1_zgl(2) in MATLAB is x1_zgl[1] in Python (0-based indexing)
    return x1_zgl[1]

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