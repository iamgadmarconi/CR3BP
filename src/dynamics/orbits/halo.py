import numpy as np

from src.dynamics.orbits.utils import _gamma_L, _find_x_crossing
from src.dynamics.stm import _compute_stm


def halo_diff_correct(x0_guess, mu, tol=1e-12, max_iter=250, solver_kwargs=None):
    """
    Diff-correction for a halo orbit in the CR3BP (CASE=1: fix z0).
    
    Parameters
    ----------
    x0_guess : array_like, shape (6,)
        Initial state guess [x0, y0, z0, vx0, vy0, vz0].
    mu : float
        Three-body mass parameter.
    tol : float, optional
        Convergence tolerance on Dx1.  (Default: 1e-12)
    max_iter : int, optional
        Maximum number of correction iterations.  (Default: 25)
    solver_kwargs : dict, optional
        Additional keyword arguments passed to the underlying ODE solver 
        in `_find_x_crossing` and `_compute_stm`.

    Returns
    -------
    XH : ndarray, shape (6,)
        Corrected state.
    TH : float
        Time at half-period crossing (when y=0 crossing is found).
    PERIOD : float
        Full estimated orbit period (2 * TH).
    """
    forward = 1
    X0 = np.copy(x0_guess)

    if solver_kwargs is None:
        solver_kwargs = {}

    # We will iterate until Dx1 is small enough
    Dx1 = 1.0
    attempt = 0

    # For convenience:
    mu2 = 1.0 - mu

    while abs(Dx1) > tol:
        if attempt > max_iter:
            raise RuntimeError("Maximum number of correction attempts exceeded.")

        # 1) Integrate until we find the x-crossing
        t1, xx1 = _find_x_crossing(X0, mu, forward=forward, **solver_kwargs)
        x1, y1, z1, Dx1, Dy1, Dz1 = xx1

        # 2) Compute the state transition matrix at that crossing
        #    x, t, phi, PHI = _compute_stm(...)
        #      - x, t are the raw trajectory & time arrays (unused below)
        #      - phi is the final fundamental solution matrix (6x6)
        #      - PHI is the full time history if needed (not used here)
        _, _, phi, _ = _compute_stm(X0, mu, t1, forward=forward, **solver_kwargs)

        # 3) Compute partial derivatives for correction
        #    (these replicate the CR3BP equations used in the Matlab code)
        rho1 = 1.0 / ((x1 + mu)**2 + y1**2 + z1**2)**1.5
        rho2 = 1.0 / ((x1 - mu2)**2 + y1**2 + z1**2)**1.5

        # second-derivatives
        omgx1 = -(mu2 * (x1 + mu) * rho1) - (mu * (x1 - mu2) * rho2) + x1
        DDz1  = -(mu2 * z1 * rho1) - (mu * z1 * rho2)
        DDx1  = 2.0 * Dy1 + omgx1

        # 4) CASE=1 Correction: fix z0
        #    We want to kill Dx1 and Dz1 by adjusting x0 and Dy0.
        #    In the Matlab code:
        #
        #    C1 = [phi(4,1) phi(4,5);
        #          phi(6,1) phi(6,5)];
        #    C2 = C1 - (1/Dy1)*[DDx1 DDz1]'*[phi(2,1) phi(2,5)];
        #    C3 = inv(C2)*[-Dx1 -Dz1]';
        #    dx0  = C3(1);
        #    dDy0 = C3(2);

        # In Python, remember phi is zero-indexed.  The original phi(i,j)
        # in Matlab is phi[i-1, j-1] in Python:
        #   phi(4,1) --> phi[3,0]
        #   phi(4,5) --> phi[3,4]
        #   phi(2,1) --> phi[1,0]   etc.
        C1 = np.array([[phi[3, 0], phi[3, 4]],
                       [phi[5, 0], phi[5, 4]]])

        # Vector for partial derivative in the (Dx, Dz) direction
        # [DDx1, DDz1]^T (2x1) times [phi(2,1), phi(2,5)] (1x2)
        # We'll reshape to keep consistent matrix operations in Python.
        dd_vec = np.array([[DDx1], [DDz1]])
        phi_2 = np.array([[phi[1, 0], phi[1, 4]]])  # shape (1,2)

        # Compute the correction matrix: 2x2 minus (1/Dy1)*(2x1 * 1x2) = 2x2
        C2 = C1 - (1.0 / Dy1) * (dd_vec @ phi_2)

        # The correction right-hand side is [-Dx1, -Dz1].
        rhs = np.array([[-Dx1], [-Dz1]])

        # Solve for [dx0, dDy0].
        delta = np.linalg.inv(C2) @ rhs
        dx0  = delta[0, 0]
        dDy0 = delta[1, 0]

        # Apply corrections to the initial guess
        X0[0] += dx0
        X0[4] += dDy0

        attempt += 1

    # After convergence, return the corrected solution:
    XH = X0
    TH = t1

    return XH, TH

def halo_orbit_ic(mu, Lpt, Azlp=0.2, n=1):
    """
    Generate initial conditions for a halo orbit near a libration point.
    
    This function computes the initial conditions for a halo orbit near a specified
    libration point (L1, L2, or L3) in the Circular Restricted Three-Body Problem (CR3BP).
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system.
    Lpt : int
        Integer (1, 2, or 3) indicating which collinear libration point to calculate:
        * 1: L1 (between the two primaries)
        * 2: L2 (beyond the smaller primary)
        * 3: L3 (beyond the larger primary)
    Azlp : float
        Distance from the libration point to the point of closest approach.
    n : int {1, -1}
        * 1: Northern Halo (z>0, Class I)
        * -1: Southern Halo (z<0, Class II)

    Returns
    -------
    x0 : array_like
        Initial conditions for the halo orbit.
    
    Notes
    -----
    The function uses the linearized dynamics around the libration point to compute
    the initial conditions. The returned state vector `x0` contains position and
    velocity components in the order [x, y, z, xdot, ydot, zdot].

    Examples
    --------
    >>> mu = 0.01215  # Earth-Moon system
    >>> Lpt = 1  # L1
    >>> Azlp = 0.05  # 5% of the distance between the two primaries
    >>> x0 = halo_orbit_ic(mu, Lpt, Azlp)
    >>> print(x0)
    """
    gamma = _gamma_L(mu, Lpt)
    Az = Azlp

    # Determine sign (won) and which "primary" to use
    if Lpt == 1:
        won = +1
        primary = 1 - mu
    elif Lpt == 2:
        won = -1
        primary = 1 - mu
    elif Lpt == 3:
        won = +1
        primary = -mu

    # Coefficients c(2), c(3), c(4)
    # We'll store them in a Python list indexed by [2,3,4]
    c = [0.0, 0.0, 0.0, 0.0, 0.0]  # just to keep 5 slots: c[2], c[3], c[4]
    
    if Lpt == 3:
        for N in [2, 3, 4]:
            c[N] = (1 / gamma**3) * (
                (1 - mu) + (-primary * gamma**(N + 1)) / ((1 + gamma)**(N + 1))
            )
    else:
        for N in [2, 3, 4]:
            c[N] = (1 / gamma**3) * (
                (won**N) * mu 
                + ((-1)**N)
                * (primary * gamma**(N + 1) / ((1 + (-won) * gamma)**(N + 1)))
            )

    # Solve for lambda (the in-plane frequency)
    polylambda = [
        1,
        0,
        c[2] - 2,
        0,
        - (c[2] - 1) * (1 + 2 * c[2]),
    ]
    lambda_roots = np.roots(polylambda)

    # The original code picks a specific root based on Lpt:
    # If Lpt == 3, pick abs(lambda(3)) => the third root in MATLAB indexing
    # else pick abs(lambda(1)) => the first root. We'll replicate that exactly.
    if Lpt == 3:
        lam = abs(lambda_roots[2])  # third element in 0-based indexing
    else:
        lam = abs(lambda_roots[0])  # first element in 0-based indexing

    # Follow the same variable definitions as in the MATLAB code
    k = 2 * lam / (lam**2 + 1 - c[2])
    delta = lam**2 - c[2]

    d1 = (3 * lam**2 / k) * (k * (6 * lam**2 - 1) - 2 * lam)
    d2 = (8 * lam**2 / k) * (k * (11 * lam**2 - 1) - 2 * lam)

    a21 = (3 * c[3] * (k**2 - 2)) / (4 * (1 + 2 * c[2]))
    a22 = (3 * c[3]) / (4 * (1 + 2 * c[2]))
    a23 = - (3 * c[3] * lam / (4 * k * d1)) * (
        3 * k**3 * lam - 6 * k * (k - lam) + 4
    )
    a24 = - (3 * c[3] * lam / (4 * k * d1)) * (2 + 3 * k * lam)

    b21 = - (3 * c[3] * lam / (2 * d1)) * (3 * k * lam - 4)
    b22 = (3 * c[3] * lam) / d1

    d21 = - c[3] / (2 * lam**2)

    a31 = (
        - (9 * lam / (4 * d2)) 
        * (4 * c[3] * (k * a23 - b21) + k * c[4] * (4 + k**2)) 
        + ((9 * lam**2 + 1 - c[2]) / (2 * d2)) 
        * (
            3 * c[3] * (2 * a23 - k * b21) 
            + c[4] * (2 + 3 * k**2)
        )
    )
    a32 = (
        - (1 / d2)
        * (
            (9 * lam / 4) * (4 * c[3] * (k * a24 - b22) + k * c[4]) 
            + 1.5 * (9 * lam**2 + 1 - c[2]) 
            * (c[3] * (k * b22 + d21 - 2 * a24) - c[4])
        )
    )

    b31 = (
        0.375 / d2
        * (
            8 * lam 
            * (3 * c[3] * (k * b21 - 2 * a23) - c[4] * (2 + 3 * k**2))
            + (9 * lam**2 + 1 + 2 * c[2])
            * (4 * c[3] * (k * a23 - b21) + k * c[4] * (4 + k**2))
        )
    )
    b32 = (
        (1 / d2)
        * (
            9 * lam 
            * (c[3] * (k * b22 + d21 - 2 * a24) - c[4])
            + 0.375 * (9 * lam**2 + 1 + 2 * c[2])
            * (4 * c[3] * (k * a24 - b22) + k * c[4])
        )
    )

    d31 = (3 / (64 * lam**2)) * (4 * c[3] * a24 + c[4])
    d32 = (3 / (64 * lam**2)) * (4 * c[3] * (a23 - d21) + c[4] * (4 + k**2))

    s1 = (
        1 
        / (2 * lam * (lam * (1 + k**2) - 2 * k))
        * (
            1.5 * c[3] 
            * (
                2 * a21 * (k**2 - 2) 
                - a23 * (k**2 + 2) 
                - 2 * k * b21
            )
            - 0.375 * c[4] * (3 * k**4 - 8 * k**2 + 8)
        )
    )
    s2 = (
        1 
        / (2 * lam * (lam * (1 + k**2) - 2 * k))
        * (
            1.5 * c[3] 
            * (
                2 * a22 * (k**2 - 2) 
                + a24 * (k**2 + 2) 
                + 2 * k * b22 
                + 5 * d21
            )
            + 0.375 * c[4] * (12 - k**2)
        )
    )

    a1 = -1.5 * c[3] * (2 * a21 + a23 + 5 * d21) - 0.375 * c[4] * (12 - k**2)
    a2 =  1.5 * c[3] * (a24 - 2 * a22) + 1.125 * c[4]

    l1 = a1 + 2 * lam**2 * s1
    l2 = a2 + 2 * lam**2 * s2

    tau1 = 0.0
    deltan = - n  # matches the original code's sign usage

    # Solve for Ax from the condition ( -del - l2*Az^2 ) / l1
    Ax = np.sqrt((-delta - l2 * Az**2) / l1)

    # Evaluate the expansions at tau1 = 0
    x = (
        a21 * Ax**2 + a22 * Az**2
        - Ax * np.cos(tau1)
        + (a23 * Ax**2 - a24 * Az**2) * np.cos(2 * tau1)
        + (a31 * Ax**3 - a32 * Ax * Az**2) * np.cos(3 * tau1)
    )
    y = (
        k * Ax * np.sin(tau1)
        + (b21 * Ax**2 - b22 * Az**2) * np.sin(2 * tau1)
        + (b31 * Ax**3 - b32 * Ax * Az**2) * np.sin(3 * tau1)
    )
    z = (
        deltan * Az * np.cos(tau1)
        + deltan * d21 * Ax * Az * (np.cos(2 * tau1) - 3)
        + deltan * (d32 * Az * Ax**2 - d31 * Az**3) * np.cos(3 * tau1)
    )

    xdot = (
        lam * Ax * np.sin(tau1)
        - 2 * lam * (a23 * Ax**2 - a24 * Az**2) * np.sin(2 * tau1)
        - 3 * lam * (a31 * Ax**3 - a32 * Ax * Az**2) * np.sin(3 * tau1)
    )
    ydot = (
        lam
        * (
            k * Ax * np.cos(tau1)
            + 2 * (b21 * Ax**2 - b22 * Az**2) * np.cos(2 * tau1)
            + 3 * (b31 * Ax**3 - b32 * Ax * Az**2) * np.cos(3 * tau1)
        )
    )
    zdot = (
        - lam * deltan * Az * np.sin(tau1)
        - 2 * lam * deltan * d21 * Ax * Az * np.sin(2 * tau1)
        - 3 * lam * deltan * (d32 * Az * Ax**2 - d31 * Az**3) * np.sin(3 * tau1)
    )

    # Scale back by gamma
    # r0 in the MATLAB code is gamma * [ (primary + gamma*(-won + x))/gamma - y z ]'.
    # That expands to:
    #   r0(1) = primary + gamma*(-won + x)
    #   r0(2) = - gamma * y
    #   r0(3) =   gamma * z
    rx = primary + gamma * (-won + x)
    ry = -gamma * y
    rz =  gamma * z

    vx = gamma * xdot
    vy = gamma * ydot
    vz = gamma * zdot

    # Combine into one 6D state vector [r0, v0]
    x0 = np.array([rx, ry, rz, vx, vy, vz], dtype=float)
    return x0