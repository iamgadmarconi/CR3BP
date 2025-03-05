import numpy as np
from scipy.integrate import solve_ivp

from dynamics.dynamics import variational_equations


def _compute_stm(x0, mu, tf, forward=1, **solve_kwargs):
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