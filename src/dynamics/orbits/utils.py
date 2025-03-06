"""
Orbital utilities for the Circular Restricted Three-Body Problem (CR3BP).

This module provides utility functions that support the computation and analysis
of periodic orbits in the CR3BP. It includes functionality for root-finding,
detecting crossings of orbits with coordinate planes, and computing ranges
for orbit families.

These utilities form the backbone for higher-level orbit computation methods,
particularly for Lyapunov and halo orbit families. The functions handle numerical
challenges like finding precise crossing times and bracketing roots in nonlinear
functions.
"""

import numpy as np
from scipy.optimize import root_scalar

from src.dynamics.propagator import propagate_crtbp


def _find_bracket(f, x0, max_expand=500):
    """
    Find a bracketing interval for a root and solve using Brent's method.
    
    This function attempts to locate an interval containing a root of the function f
    by expanding outward from an initial guess x0. Once a sign change is detected,
    it applies Brent's method to find the precise root location. The approach is
    similar to MATLAB's fzero function.
    
    Parameters
    ----------
    f : callable
        The function for which we want to find a root f(x)=0
    x0 : float
        Initial guess for the root location
    max_expand : int, optional
        Maximum number of expansion iterations to try.
        Default is 500.
    
    Returns
    -------
    float
        The location of the root, as determined by Brent's method
    
    Notes
    -----
    The function works by:
    1. Starting with a small step size (1e-10)
    2. Testing points on both sides of x0 (x0±dx)
    3. If a sign change is detected, applying Brent's method to find the root
    4. If no sign change is found, increasing the step size by sqrt(2) and trying again
    
    This approach is effective for finding roots of smooth functions where a
    reasonable initial guess is available, particularly for orbital period and
    crossing time calculations.
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

def _find_x_crossing(x0, mu, forward=1, **solver_kwargs):
    """
    Find the time and state at which an orbit next crosses the y=0 plane.
    
    This function propagates a trajectory from an initial state and determines
    when it next crosses the y=0 plane (i.e., the x-z plane). This is particularly
    useful for periodic orbit computations where the orbit is symmetric about the
    x-z plane.
    
    Parameters
    ----------
    x0 : array_like
        Initial state vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    forward : {1, -1}, optional
        Direction of time integration:
        * 1: forward in time (default)
        * -1: backward in time
    
    Returns
    -------
    t1_z : float
        Time at which the orbit crosses the y=0 plane
    x1_z : ndarray
        State vector [x, y, z, vx, vy, vz] at the crossing
    
    Notes
    -----
    The function uses a two-step approach:
    1. First integrating to an approximate time where the crossing is expected (π/2 - 0.15)
    2. Then using a root-finding method to precisely locate the crossing time
    
    This hybrid approach is more efficient than using a very fine integration
    time step, especially for orbits with long periods.
    """
    tolzero = 1.e-10
    # Initial guess for the time
    t0_z = np.pi/2 - 0.15

    # 1) Integrate from t=0 up to t0_z.
    sol = propagate_crtbp(x0, 0.0, t0_z, mu, forward=forward, steps=500, **solver_kwargs)
    xx = sol.y.T  # assume sol.y is (state_dim, time_points)
    x0_z = xx[-1]  # final state after integration

    # 2) Define a local function that depends on time t.
    def halo_y_wrapper(t):
        """
        Wrapper function that returns the y-coordinate of the orbit at time t.
        
        This function is used as the target for root-finding, since we want to
        find where y=0 (i.e., when the orbit crosses the x-z plane).
        
        Parameters
        ----------
        t : float
            Time to evaluate the orbit
        
        Returns
        -------
        float
            The y-coordinate of the orbit at time t
        """
        return _halo_y(t, t0_z, x0_z, mu, forward=forward, steps=500)

    # 3) Find the time at which y=0 by bracketing the root.
    t1_z = _find_bracket(halo_y_wrapper, t0_z)

    # 4) Integrate from t0_z to t1_z to get the final state.
    sol = propagate_crtbp(x0_z, t0_z, t1_z, mu, forward=forward, steps=500, **solver_kwargs)
    xx_final = sol.y.T
    x1_z = xx_final[-1]

    return t1_z, x1_z


def _halo_y(t1, t0_z, x0_z, mu, forward=1, steps=3000, tol=1e-10):
    """
    Compute the y-component of an orbit at a specified time.
    
    This function propagates an orbit from a reference state and time to a
    target time, and returns the y-component of the resulting state. It is
    designed to be used for finding orbital plane crossings.
    
    Parameters
    ----------
    t1 : float
        Target time at which to evaluate the y-component
    t0_z : float
        Reference time corresponding to the reference state x0_z
    x0_z : array_like
        Reference state vector [x, y, z, vx, vy, vz] at time t0_z
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    forward : {1, -1}, optional
        Direction of time integration:
        * 1: forward in time (default)
        * -1: backward in time
    steps : int, optional
        Number of integration steps to use. Default is 3000.
    tol : float, optional
        Tolerance for the numerical integrator. Default is 1e-10.
    
    Returns
    -------
    float
        The y-component of the orbit state at time t1
    
    Notes
    -----
    This function is primarily used within root-finding algorithms to locate
    precise times when the orbit crosses the y=0 plane. It avoids unnecessary
    computation when t1 is very close to t0_z by simply returning the y-component
    of the reference state in that case.
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

def _z_range(L_i, x0i):
    """
    Compute the range of z-coordinates for generating a Halo orbit family.
    
    This function determines an appropriate range of z-coordinates to use when
    computing a family of Halo orbits, based on the location of the libration
    point and the initial orbit.
    
    Parameters
    ----------
    L_i : array_like
        Coordinates of the libration point [x, y, z] in the rotating frame
    x0i : array_like
        Initial state vector [x, y, z, vx, vy, vz] for the starting orbit
    
    Returns
    -------
    tuple
        A tuple (zmin, zmax) defining the range of z-values to use
    
    Notes
    -----
    The minimum z-value is set relative to the libration point, while the
    maximum is fixed at 0.05 (in dimensionless units). This range typically
    captures a comprehensive family of Halo orbits from small to large
    amplitudes.
    """
    z_candidates = [L_i[2], x0i[2]]
    zmin = min(z_candidates)
    zmax = max(z_candidates)
    return zmin, zmax

def _x_range(L_i, x0i):
    """
    Compute the range of x-values for generating a Lyapunov orbit family.
    
    This function determines an appropriate range of x-coordinates to use when
    computing a family of Lyapunov orbits, based on the location of the libration
    point and the initial orbit.
    
    Parameters
    ----------
    L_i : array_like
        Coordinates of the libration point [x, y, z] in the rotating frame
    x0i : array_like
        Initial state vector [x, y, z, vx, vy, vz] for the starting orbit
    
    Returns
    -------
    tuple
        A tuple (xmin, xmax) defining the range of x-values to use
    
    Notes
    -----
    The minimum x-value is set relative to the libration point, while the
    maximum is fixed at 0.05 (in dimensionless units). This range typically
    captures a comprehensive family of Lyapunov orbits from small to large
    amplitudes.
    """
    xmin = x0i[0] - L_i[0]
    xmax = 0.05
    return xmin, xmax

def _gamma_L(mu, Lpt):
    """
    Calculate the ratio of libration point distance from the closest primary to the distance
    between the two primaries in the CR3BP.
    
    This function computes the normalized distance (gamma) between a collinear libration point 
    (L1, L2, or L3) and its nearest primary body in the Circular Restricted Three-Body Problem.
    The value is expressed as a ratio of the distance between the two primary bodies.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system, defined as the ratio of the smaller primary's 
        mass to the total system mass (0 < mu < 1). For example, in the Sun-Earth system, 
        mu ≈ 3.0034e-6.
    Lpt : int
        Integer (1, 2, or 3) indicating which collinear libration point to calculate:
        * 1: L1 (between the two primaries)
        * 2: L2 (beyond the smaller primary)
        * 3: L3 (beyond the larger primary)
    
    Returns
    -------
    float
        The gamma ratio for the specified libration point. This represents the normalized
        distance between the libration point and its nearest primary body.
    
    Notes
    -----
    The function solves the quintic polynomials that determine the positions of the
    collinear libration points L1, L2, and L3. These are derived from the equations
    of motion in the CR3BP.
    
    The gamma values have the following interpretations:
    - For L1: gamma is the distance from the smaller primary to L1, toward the larger primary
    - For L2: gamma is the distance from the smaller primary to L2, away from the larger primary
    - For L3: gamma is the distance from the larger primary to L3, away from the smaller primary
    
    All distances are normalized by the distance between the two primaries.
    
    The polynomials solved are:
    - L1: x^5 - (3-μ)x^4 + (3-2μ)x^3 - μx^2 + 2μx - μ = 0
    - L2: x^5 + (3-μ)x^4 + (3-2μ)x^3 - μx^2 - 2μx - μ = 0
    - L3: x^5 + (2+μ)x^4 + (1+2μ)x^3 - (1-μ)x^2 - 2(1-μ)x - (1-μ) = 0
    
    Raises
    ------
    ValueError
        If Lpt is not 1, 2, or 3, or if no real root is found for the polynomial.
    
    Examples
    --------
    >>> # Calculate gamma for L1 in the Earth-Moon system (mu ≈ 0.01215)
    >>> gamma_l1 = _gamma_L(0.01215, 1)
    >>> print(f"L1 is approximately {gamma_l1:.6f} of the Earth-Moon distance from the Moon")
    """
    mu2 = 1 - mu

    # Define polynomial coefficients as in the MATLAB code:
    poly1 = [1, -1*(3-mu), (3-2*mu), -mu, 2*mu, -mu]
    poly2 = [1, (3-mu), (3-2*mu), -mu, -2*mu, -mu]
    poly3 = [1, (2+mu), (1+2*mu), -mu2, -2*mu2, -mu2]

    # Compute roots
    rt1 = np.roots(poly1)
    rt2 = np.roots(poly2)
    rt3 = np.roots(poly3)

    # Find the last real root for each polynomial
    GAMMAS = [None, None, None]
    for r in rt1:
        if np.isreal(r):
            GAMMAS[0] = r.real
    for r in rt2:
        if np.isreal(r):
            GAMMAS[1] = r.real
    for r in rt3:
        if np.isreal(r):
            GAMMAS[2] = r.real

    return GAMMAS[Lpt-1]