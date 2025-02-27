import numpy as np

from utils.frames import (_libration_frame_eigenvectors, 
                        _mu_bar, _alpha_1, _alpha_2, _beta_1, _beta_2)

from .crtbp import crtbp_energy
from .corrector import lyapunov_diff_correct

def general_linear_ic(mu, L_i):
    """
    Returns the rotating-frame initial condition for the linearized CR3BP
    near the libration point L_i, for the chosen coefficients alpha1, alpha2, beta1, beta2.
    """
    # 1) Get the four eigenvectors (u1,u2,w1,w2)
    u1, u2, w1, w2 = _libration_frame_eigenvectors(mu, L_i)
    alpha1 = _alpha_1(mu, L_i)
    alpha2 = _alpha_2(mu, L_i)
    beta1 = _beta_1(mu, L_i)
    beta2 = _beta_2(mu, L_i)

    beta = beta1 + beta2*1j

    term = 2 * np.real(beta * w1)

    z = alpha1 * u1 + alpha2 * u2 + term
    x, y, vx, vy = z[0], z[1], z[2], z[3]

    return np.array([x, y, 0, vx, vy, 0], dtype=np.float64)

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


def lyapunov_family(mu, L_i, x0i, dx=0.0001):
    """
    Generate a family of planar Lyapunov orbits about L1 (or L2) by 
    numerical continuation on the initial x-position.

    Parameters
    ----------
    mu  : float
        3-Body mass ratio
    xmin, xmax : float
        Range of x-values over which to continue the family
    dx   : float
        Step in x between consecutive orbits
    x0i  : array-like, shape (6,)
        Initial guess of 6D state for the first orbit in the family
    Returns
    -------
    xL  : ndarray, shape (N, 6)
        Array of all corrected initial conditions for the family
    t1L : ndarray, shape (N,)
        Array of half-periods corresponding to each orbit
    """

    # Build the range of x-values
    xmin, xmax = _x_range(L_i, x0i)
    x_vals = np.arange(xmin, xmax, dx)
    # Storage
    xL_list  = []
    t1L_list = []

    # Start from the user-provided initial guess
    x0_current = np.array(x0i, copy=True)

    for x_desired in x_vals:
        # Modify x0_current(0) to the new x
        x0_current[0] = x_desired
        
        # Use your differential corrector
        x0_corrected, t_half = lyapunov_diff_correct(x0_current, mu)
        
        # Store in the family
        xL_list.append(x0_corrected)
        t1L_list.append(t_half)
        
        # Update guess for the next step
        x0_current = x0_corrected
    
    # Convert to numpy arrays
    xL  = np.array(xL_list)
    t1L = np.array(t1L_list)
    
    return xL, t1L


def _x_range(L_i, x0i):
    """
    Returns the range of x-values for the Lyapunov family.
    """
    xmin = x0i[0] - L_i[0]
    xmax = 0.05
    return xmin, xmax
