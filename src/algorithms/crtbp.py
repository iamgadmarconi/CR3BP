"""
Circular Restricted Three-Body Problem (CR3BP) core functions.

This module provides the fundamental mathematical tools for analyzing the 
Circular Restricted Three-Body Problem (CR3BP). It includes functions for:

1. Computing energy and Jacobi constants of orbits
2. Locating and analyzing libration (Lagrange) points
3. Computing distances to the primary and secondary bodies
4. Calculating potential functions and effective potentials
5. Determining Hill regions (forbidden regions) in the CR3BP
6. Analyzing energy bounds for different regimes of motion

These tools form the foundation for understanding the dynamics of the CR3BP
and provide the basis for more advanced analyses like periodic orbits and
invariant manifolds.
"""

import numba
import math
import mpmath as mp
import numpy as np
import warnings

mp.mp.dps = 50  # Set mpmath precision to 50 digits


def _libration_index_to_coordinates(mu, L_i):
    """
    Convert libration point index to coordinates.

    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    L_i : int
        Libration point index (1-5)

    Returns
    -------
    ndarray
        3D vector [x, y, z] giving the position of the libration point

    Notes
    -----
    The libration points are located at the equilibrium points of the CR3BP.
    """
    if L_i == 1:
        return _l1(mu)
    elif L_i == 2:
        return _l2(mu)
    elif L_i == 3:
        warnings.warn("Logic for L3 is not implemented, proceed with caution")
        return _l3(mu)
    elif L_i == 4:
        warnings.warn("Logic for L4 is not implemented, proceed with caution")
        return _l4(mu)
    elif L_i == 5:
        warnings.warn("Logic for L5 is not implemented, proceed with caution")
        return _l5(mu)
    else:
        raise ValueError("Invalid libration point index")

def crtbp_energy(state, mu):
    """
    Compute the energy (Hamiltonian) of a state in the CR3BP.
    
    This function calculates the total energy of a given state in the CR3BP,
    which is a conserved quantity in the rotating frame and is related to
    the Jacobi constant by C = -2E.
    
    Parameters
    ----------
    state : array_like
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    float
        The energy value (scalar)
    
    Notes
    -----
    The energy in the rotating frame consists of the kinetic energy plus
    the effective potential, which includes the gravitational potential and
    the centrifugal potential. This is a conserved quantity along any trajectory
    in the CR3BP.
    """
    x, y, z, vx, vy, vz = state
    mu1 = 1.0 - mu
    mu2 = mu
    
    r1 = np.sqrt((x + mu2)**2 + y**2 + z**2)
    r2 = np.sqrt((x - mu1)**2 + y**2 + z**2)
    
    kin = 0.5 * (vx*vx + vy*vy + vz*vz)
    pot = -(mu1 / r1) - (mu2 / r2) - 0.5*(x*x + y*y + z*z) - 0.5*mu1*mu2
    return kin + pot

def compute_energy_bounds(mu, case):
    """
    Compute the energy bounds for different regimes of motion in the CR3BP.
    
    This function determines the energy boundaries that separate qualitatively
    different regimes of motion in the CR3BP, based on the energy levels at
    the libration points.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    case : int
        Case number (1-5) indicating which energy regime to compute:
        * Case 1: Below L1 energy - motion bound near one primary only
        * Case 2: Between L1 and L2 energy - L1 neck open, others closed
        * Case 3: Between L2 and L3 energy - L1 and L2 open, L3 closed
        * Case 4: Between L3 and L4/L5 energy - all collinear passages open
        * Case 5: At or above L4/L5 energy - motion virtually unrestricted
    
    Returns
    -------
    tuple
        A tuple (E_lower, E_upper) giving the energy range for the specified case
    
    Raises
    ------
    ValueError
        If mu is outside the range [0, 0.5] or case is not between 1 and 5
    
    Notes
    -----
    The energy levels at the libration points create natural boundaries that
    determine which regions of space are accessible to a given orbit. This
    function computes these boundaries, which are essential for understanding
    the global dynamics in the CR3BP.
    """
    if mu < 0 or mu > 0.5:
        raise ValueError("Mass ratio mu must be between 0 and 0.5 (inclusive).")

    if abs(mu) < 1e-9:  # treat mu == 0 as two-body problem (secondary has zero mass)
        E1 = E2 = E3 = E4 = E5 = -1.5
    else:

        x_L1 = _l1(mu)
        x_L2 = _l2(mu)
        x_L3 = _l3(mu)

        def Omega(x, y, mu):
            """
            Compute the effective potential at a point (x,y) in the rotating frame.
            
            Parameters
            ----------
            x : float
                x-coordinate in the rotating frame
            y : float
                y-coordinate in the rotating frame
            mu : float
                Mass parameter of the CR3BP
            
            Returns
            -------
            float
                The effective potential value
            """
            r1 = np.sqrt((x + mu)**2 + y**2)
            r2 = np.sqrt((x - 1 + mu)**2 + y**2)
            return 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2
        
        E1 = -Omega(x_L1[0], 0.0, mu)   # energy level at L1
        E2 = -Omega(x_L2[0], 0.0, mu)   # energy level at L2
        E3 = -Omega(x_L3[0], 0.0, mu)   # energy level at L3
        
        E4 = E5 = -1.5
    
    if case == 1:
        return (-math.inf, E1)
    elif case == 2:
        return (E1, E2)
    elif case == 3:
        return (E2, E3)
    elif case == 4:
        return (E3, E4)
    elif case == 5:
        return (E4, math.inf)
    else:
        raise ValueError("Case number must be between 1 and 5.")

def _energy_to_jacobi_constant(E):
    """
    Convert energy to Jacobi constant.
    
    The Jacobi constant C is related to the energy E by C = -2E.
    
    Parameters
    ----------
    E : float
        Energy value
    
    Returns
    -------
    float
        Corresponding Jacobi constant
    """
    return -2 * E

def hill_region(mu, C, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), n_grid=400):
    """
    Compute the Hill region (zero-velocity curves) for a given Jacobi constant.
    
    This function calculates the regions in the x-y plane that are accessible
    to an orbit with a specific Jacobi constant. The boundaries of these regions
    are the zero-velocity curves where the kinetic energy is exactly zero.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    C : float
        Jacobi constant value
    x_range : tuple, optional
        Range of x values to compute (xmin, xmax). Default is (-1.5, 1.5).
    y_range : tuple, optional
        Range of y values to compute (ymin, ymax). Default is (-1.5, 1.5).
    n_grid : int, optional
        Number of grid points in each dimension. Default is 400.
    
    Returns
    -------
    X : ndarray
        2D array of x-coordinates
    Y : ndarray
        2D array of y-coordinates
    Z : ndarray
        2D array of values where Z > 0 indicates forbidden regions and
        Z ≤ 0 indicates allowed regions
    
    Notes
    -----
    The Hill region calculation is a powerful tool for visualizing the
    accessible regions of phase space. Points where Z > 0 are inaccessible
    (forbidden) to an orbit with the given Jacobi constant, while points
    where Z ≤ 0 are accessible.
    """
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(x, y)

    r1 = np.sqrt((X + mu)**2 + Y**2)
    r2 = np.sqrt((X - 1 + mu)**2 + Y**2)

    Omega = (1 - mu) / r1 + mu / r2 + 0.5 * (X**2 + Y**2)

    Z = Omega - C/2

    return X, Y, Z

def _kinetic_energy(state):
    """
    Compute the kinetic energy of a state.
    
    Parameters
    ----------
    state : array_like
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    
    Returns
    -------
    float
        Kinetic energy value
    """
    x, y, z, vx, vy, vz = state
    return 1 / 2 * (vx**2 + vy**2 + vz**2)

def energy_integral(state, mu):
    """
    Compute the energy integral (Hamiltonian) of a state in the CR3BP.
    
    This function calculates the total energy of a given state, which
    is a conserved quantity along any trajectory in the CR3BP.
    
    Parameters
    ----------
    state : array_like
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    float
        The energy value
    
    Notes
    -----
    This is an alternative implementation of the energy calculation that
    uses the effective potential explicitly. It should give the same result
    as crtbp_energy().
    """
    x, y, z, vx, vy, vz = state
    U_eff = _effective_potential(state, mu)
    return 1 / 2 * (vx**2 + vy**2 + vz**2) + U_eff

def jacobi_constant(state, mu):
    """
    Compute the Jacobi constant of a state in the CR3BP.
    
    The Jacobi constant is an important conserved quantity in the CR3BP
    that helps classify and analyze different types of orbits.
    
    Parameters
    ----------
    state : array_like
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    float
        The Jacobi constant value
    
    Notes
    -----
    The Jacobi constant C is related to the energy E by C = -2E.
    It remains constant along any trajectory in the CR3BP and is
    often used to characterize different regimes of motion.
    """
    x, y, z, vx, vy, vz = state
    U_eff = _effective_potential(state, mu)
    return - (vx**2 + vy**2 + vz**2) - 2 * U_eff

def _effective_potential(state, mu):
    """
    Compute the effective potential of a state in the CR3BP.
    
    The effective potential includes both the gravitational potential
    and the centrifugal potential in the rotating frame.
    
    Parameters
    ----------
    state : array_like
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    float
        The effective potential value
    
    Notes
    -----
    The effective potential is the sum of the gravitational potential
    and the centrifugal potential. It determines the shape of the
    zero-velocity curves and the location of the libration points.
    """
    x, y, z, vx, vy, vz = state
    mu_1 = 1 - mu
    mu_2 = mu
    r1 = _primary_distance(state, mu)
    r2 = _secondary_distance(state, mu)
    U = _potential(state, mu)
    U_eff = - 1 / 2 * (x**2 + y**2 + z**2) + U
    return U_eff

def _potential(state, mu):
    """
    Compute the gravitational potential of a state in the CR3BP.
    
    This function calculates the gravitational potential energy due
    to the two primary bodies in the CR3BP.
    
    Parameters
    ----------
    state : array_like
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    float
        The gravitational potential value
    """
    x, y, z, vx, vy, vz = state
    mu_1 = 1 - mu
    mu_2 = mu
    r1 = _primary_distance(state, mu)
    r2 = _secondary_distance(state, mu)
    U = - mu_1 / r1 - mu_2 / r2 - 1 / 2 * mu_1 * mu_2
    return U

def _primary_distance(state, mu):
    """
    Compute the distance from a state to the primary body.
    
    Parameters
    ----------
    state : array_like
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    float
        Distance to the primary body
    """
    x, y, z, vx, vy, vz = state
    mu_2 = mu
    r1 = np.sqrt((x + mu_2)**2 + y**2 + z**2)
    return r1

def _secondary_distance(state, mu):
    """
    Compute the distance from a state to the secondary body.
    
    Parameters
    ----------
    state : array_like
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    float
        Distance to the secondary body
    """
    x, y, z, vx, vy, vz = state
    mu_1 = 1 - mu
    r2 = np.sqrt((x - mu_1)**2 + y**2 + z**2)
    return r2

def libration_points(mu):
    """
    Compute all five libration points in the CR3BP.
    
    This function calculates the positions of the five Lagrange (libration)
    points in the CR3BP for the given mass parameter.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    tuple
        A tuple containing the positions of L1, L2, L3, L4, and L5 as ndarrays
    
    Notes
    -----
    The libration points are equilibrium points in the rotating frame where
    the gravitational and centrifugal forces balance. There are three collinear
    points (L1, L2, L3) located on the x-axis, and two equilateral points
    (L4, L5) forming equilateral triangles with the primary bodies.
    """
    l1, l2, l3 = _collinear_points(mu)
    l4, l5 = _equilateral_points(mu)
    return l1, l2, l3, l4, l5

def _equilateral_points(mu):
    """
    Compute the equilateral libration points (L4, L5) in the CR3BP.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    tuple
        A tuple containing the positions of L4 and L5 as ndarrays
    """
    return _l4(mu), _l5(mu)

def _collinear_points(mu):
    """
    Compute the collinear libration points (L1, L2, L3) in the CR3BP.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    tuple
        A tuple containing the positions of L1, L2, and L3 as ndarrays
    """
    return _l1(mu), _l2(mu), _l3(mu)

def _l1(mu):
    """
    Compute the position of the L1 libration point.
    
    L1 is located between the two primary bodies.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    ndarray
        3D vector [x, 0, 0] giving the position of L1
    """
    x = mp.findroot(lambda x: _dOmega_dx(x, mu), [-mu + 0.01, 1 - mu - 0.01])
    x = float(x)
    return np.array([x, 0, 0], dtype=np.float64)

def _l2(mu):
    """
    Compute the position of the L2 libration point.
    
    L2 is located beyond the smaller primary body.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    ndarray
        3D vector [x, 0, 0] giving the position of L2
    """
    x = mp.findroot(lambda x: _dOmega_dx(x, mu), [1.0, 2.0])
    x = float(x)
    return np.array([x, 0, 0], dtype=np.float64)

def _l3(mu):
    """
    Compute the position of the L3 libration point.
    
    L3 is located beyond the larger primary body.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    ndarray
        3D vector [x, 0, 0] giving the position of L3
    """
    x = mp.findroot(lambda x: _dOmega_dx(x, mu), [-mu - 0.01, -2.0])
    x = float(x)
    return np.array([x, 0, 0], dtype=np.float64)

def _l4(mu):
    """
    Compute the position of the L4 libration point.
    
    L4 forms an equilateral triangle with the two primary bodies,
    located above the x-axis (positive y).
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    ndarray
        3D vector [x, y, 0] giving the position of L4
    """
    x = 1 / 2 - mu
    y = np.sqrt(3) / 2
    return np.array([x, y, 0], dtype=np.float64)

def _l5(mu):
    """
    Compute the position of the L5 libration point.
    
    L5 forms an equilateral triangle with the two primary bodies,
    located below the x-axis (negative y).
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    ndarray
        3D vector [x, y, 0] giving the position of L5
    """
    x = 1 / 2 - mu
    y = -np.sqrt(3) / 2
    return np.array([x, y, 0], dtype=np.float64)

def _dOmega_dx(x, mu):
    """
    Compute the derivative of the effective potential with respect to x.
    
    This function is used to find the locations of the collinear libration
    points, which occur where this derivative is zero.
    
    Parameters
    ----------
    x : float
        x-coordinate in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    float
        Value of dΩ/dx at the given x-coordinate
    """
    r1 = abs(x + mu)
    r2 = abs(x - (1 - mu))
    return x - (1 - mu) * (x + mu) / (r1**3)  -  mu * (x - (1 - mu)) / (r2**3)