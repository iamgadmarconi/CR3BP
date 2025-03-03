import numba
import numpy as np
from scipy.integrate import solve_ivp


@numba.njit(fastmath=True, cache=True)
def crtbp_accel(state, mu):
    """
    State = [x, y, z, vx, vy, vz]
    Returns the time derivative of the state vector for the CRTBP.
    """
    x, y, z, vx, vy, vz = state

    # Distances to each primary
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)      # from m1 at (-mu, 0, 0)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2) # from m2 at (1-mu, 0, 0)

    # Accelerations
    ax = 2*vy + x - (1 - mu)*(x + mu) / r1**3 - mu*(x - 1 + mu) / r2**3
    ay = -2*vx + y - (1 - mu)*y / r1**3          - mu*y / r2**3
    az = -(1 - mu)*z / r1**3 - mu*z / r2**3

    return np.array([vx, vy, vz, ax, ay, az], dtype=np.float64)

def propagate_crtbp(state0, mu, T, forward=1, steps=1000):
    """
    Integrates the CRTBP from t=0 to t=T (in absolute value), 
    but multiplies the derivatives and (optionally) the time array by `forward`.
    
    :param state0:  Initial state (x0, y0, z0, vx0, vy0, vz0)
    :param mu:      Gravitational parameter for CRTBP (mass ratio)
    :param T:       Total duration (positive number)
    :param forward: +1 for forward integration, -1 for backward integration
    :param steps:   Number of time steps for output
    :return:        An object with fields:
                      - t:  time array (may be reversed if forward=-1)
                      - y:  solution states at each time
                      - other info from solve_ivp
    """
    # We'll integrate from 0 to T in the solver. 
    # If forward=-1, we multiply the ODE by -1 inside the ODE function.
    # That way, we are effectively going backward in time.

    def ode_func(t, y):
        return forward * crtbp_accel(y, mu)

    t_span = [0, T]
    t_eval = np.linspace(0, T, steps)

    sol = solve_ivp(
        ode_func,
        t_span,
        state0,
        t_eval=t_eval,
        rtol=3e-14,
        atol=1e-14,
    )

    # If forward = -1, the ODE was integrated from 0 to +T, but 
    # each step is going backward in state. 
    # If you also want the "time" array to appear as going from 0 down to -T,
    # you can flip the sign of sol.t here:
    if forward == -1:
        sol.t = -sol.t  # so times run [0, -T], matching MATLAB's approach

    return sol
