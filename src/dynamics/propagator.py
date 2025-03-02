import numba
import numpy as np
from scipy.integrate import solve_ivp


@numba.njit(fastmath=True, cache=True)
def crtbp_accel(state, mu):
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    
    ax = 2*vy + x - (1 - mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3
    ay = -2*vx + y - (1 - mu)*y/r1**3       - mu*y/r2**3
    az = -(1 - mu)*z/r1**3 - mu*z/r2**3

    return np.array([vx, vy, vz, ax, ay, az], dtype=np.float64)


def propagate_crtbp(state0, mu, T, steps=1000):
    t_span = [0, T]
    t_eval = np.linspace(0, T, steps)
    sol = solve_ivp(
        lambda t, y: crtbp_accel(y, mu),
        t_span,
        state0,
        t_eval=t_eval,
        rtol=3e-14,
        atol=1e-14,
    )
    return sol