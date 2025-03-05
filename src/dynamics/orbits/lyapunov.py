import numpy as np
from tqdm import tqdm

from src.dynamics.propagator import propagate_crtbp
from src.dynamics.orbits.utils import _find_x_crossing, _x_range
from src.dynamics.manifolds.math import _libration_frame_eigenvectors
from scipy.integrate import solve_ivp
from src.dynamics.dynamics import variational_equations


def lyapunov_orbit_ic(mu, L_i, Ax=1e-5):
    """
    Returns the rotating-frame initial condition for the Lyapunov orbit
    near the libration point L_i.
    """
    u1, u2, u, v = _libration_frame_eigenvectors(mu, L_i, orbit_type="lyapunov")
    displacement = Ax * u

    x_L_i = L_i[0]

    state = np.array([x_L_i, 0, 0, 0], dtype=np.float64) + displacement

    x, y, vx, vy = state[0], state[1], state[2], state[3]

    return np.array([x, y, 0, vx, vy, 0], dtype=np.float64)


def lyapunov_family(mu, L_i, x0i, dx=0.0001, forward=1, max_iter=250, tol=1e-12, save=False):
    """
    Generate a family of 3D Lyapunov/Halo-like orbits by stepping x0 from xmin to xmax.
    
    We fix x0 and use a differential corrector on z0, vy0 so that the half-period crossing 
    has vx=0, vz=0 at y=0. This is akin to 'CASE=2' logic from your MATLAB code.

    Args:
      mu       : CR3BP mass ratio
      dx       : increment in x-amplitude
      x0_seed  : optional 6-vector guess [x, 0, z, vx, vy, vz]. If not given, a basic guess is used.
      max_iter : maximum attempts in the corrector

    Returns:
      xL   : shape (N, 6). Each row is a 3D initial condition in rotating coords.
      t1L  : shape (N,). The half-period for each orbit
    """
    xmin, xmax = _x_range(L_i, x0i)
    n = int(np.floor((xmax - xmin)/dx + 1))
    
    xL = []
    t1L = []

    # 1) Generate & store the first orbit
    x0_corr, t1 = lyapunov_diff_correct(x0i, mu, forward=forward, max_iter=max_iter, tol=tol)
    xL.append(x0_corr)
    t1L.append(t1)

    # 2) Step through the rest with a progress bar
    for j in tqdm(range(1, n), desc="Lyapunov family"):
        x_guess = np.copy(xL[-1])
        x_guess[0] += dx  # increment the x0 amplitude
        x0_corr, t1 = lyapunov_diff_correct(x_guess, mu, forward=forward, max_iter=max_iter, tol=tol)
        xL.append(x0_corr)
        t1L.append(t1)

    if save:
        np.save(r"src\models\xL.npy", np.array(xL, dtype=np.float64))
        np.save(r"src\models\t1L.npy", np.array(t1L, dtype=np.float64))

    return np.array(xL, dtype=np.float64), np.array(t1L, dtype=np.float64)

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

        t_cross, X_cross = _find_x_crossing(x0, mu, forward=forward)
        
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