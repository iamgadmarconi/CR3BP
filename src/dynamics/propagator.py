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

def propagate_crtbp(state0, t0, tf, mu, forward=1, steps=1000, **solve_kwargs):
    """
    Replicate MATLAB's 'FORWARD' convention.

    MATLAB's 'int()' uses a positive [0, tf] in ode45,
    multiplies the ODE by FORWARD for reverse-time,
    and then multiplies the final time array by FORWARD.
    """
    # 1) Always make the integration span positive, even if tf is negative
    t0 = abs(t0)
    tf = abs(tf)
    t_eval = np.linspace(t0, tf, steps)
    # 2) ODE function includes the forward sign, exactly like 'xdot = FORWARD * xdot'
    def ode_func(t, y):
        return forward * crtbp_accel(y, mu)

    # 4) Default tolerances, or user can override via solve_kwargs
    if 'rtol' not in solve_kwargs:
        solve_kwargs['rtol'] = 3e-14
    if 'atol' not in solve_kwargs:
        solve_kwargs['atol'] = 1e-14

    # 5) Integrate
    sol = solve_ivp(
        ode_func,
        [t0, tf],
        state0,
        t_eval=t_eval,
        **solve_kwargs
    )

    # 6) Finally, flip the reported times so that if forward = -1,
    #    the time array goes from 0 down to -T (like MATLAB's t=FORWARD*t)
    sol.t = forward * sol.t

    return sol
