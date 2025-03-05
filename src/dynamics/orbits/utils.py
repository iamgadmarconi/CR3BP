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

def _find_x_crossing(x0, mu, forward=1):
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


def _halo_y(t1, t0_z, x0_z, mu, forward=1, steps=3000, tol=1e-10):
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


