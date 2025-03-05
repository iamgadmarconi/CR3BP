import numba
import numpy as np
from scipy.integrate import solve_ivp

from dynamics.dynamics import crtbp_accel, variational_equations


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


def propagate_variational_equations(PHI_init, mu, T, forward=1, steps=1000, **solve_kwargs):
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
                    **solve_kwargs)

    # If you want the time array to reflect backward integration for forward=-1:
    if forward == -1:
        sol.t = -sol.t  # times run 0 -> -T, matching MATLAB's "t=FORWARD*t" logic

    return sol
