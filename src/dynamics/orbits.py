import numpy as np

from utils.frames import (_libration_frame_eigenvectors, 
                        _mu_bar, _alpha_1, _alpha_2, _beta_1, _beta_2)

from .crtbp import crtbp_energy
from .corrector import lyapunov_diff_correct, halo_diff_correct

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


def lyapunov_family(mu, L_i, x0i, dx=0.0001, max_iter=250):
    """
    Generate a family of 3D Lyapunov/Halo-like orbits by stepping x0 from xmin to xmax.

    We fix x0 and use a differential corrector on z0, vy0 so that the half-period crossing 
    has vx=0, vz=0 at y=0. This is akin to 'CASE=2' logic from your MATLAB code.

    Args:
      mu       : CR3BP mass ratio
      xmin     : smallest x-amplitude
      xmax     : largest x-amplitude
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
    x0_corr, t1 = lyapunov_diff_correct(x0i, mu, max_iter=max_iter)
    xL.append(x0_corr)
    t1L.append(t1)

    # 2) Step through the rest
    for j in range(1, n):
        x_guess = np.copy(xL[-1])
        x_guess[0] += dx  # increment the x0 amplitude
        x0_corr, t1 = lyapunov_diff_correct(x_guess, mu, max_iter=max_iter)
        xL.append(x0_corr)
        t1L.append(t1)

    return np.array(xL), np.array(t1L)


def halo_family(mu, L_i, x0i, dx=0.0001, max_iter=25):
    """
    Example: step x0 from xmin to xmax in increments dx,
    each time calling halo_diff_correct_3d_fixx() to fix x0 and solve for z0, vy0.
    
    x0_seed is a 6-vector guess: [x, 0, z, vx, vy, vz].
    If none provided, we create a simple guess at x=xmin.
    
    Returns (xL, t1L):
      xL shape => (N, 6) each row is an initial condition in 3D rotating coords
      t1L shape => (N,) half-period for each orbit
    """
    # Figure out how many orbits
    xmin, xmax = _x_range(L_i, x0i)

    n = int(np.floor((xmax - xmin)/dx + 1))
    
    xL = []
    t1L = []
    
    # First orbit
    x0_corrected, t1 = halo_diff_correct(x0i, mu, max_iter=max_iter)
    xL.append(x0_corrected)
    t1L.append(t1)
    
    # Loop from 2..N
    for j in range(1, n):
        fraction = j / n
        # Start from last corrected orbit
        x_guess = np.copy(xL[-1])
        # Increase x0 by dx
        x_guess[0] += dx  # fix x0 amplitude
        # Correct
        x0_corr, t1c = halo_diff_correct(x_guess, mu, max_iter=max_iter)
        xL.append(x0_corr)
        t1L.append(t1c)
    
    return np.array(xL), np.array(t1L)

def _x_range(L_i, x0i):
    """
    Returns the range of x-values for the Lyapunov family.
    """
    xmin = x0i[0] - L_i[0]
    xmax = 0.05
    return xmin, xmax
