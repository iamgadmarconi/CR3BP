"""
Energy computation functions for the Circular Restricted Three-Body Problem (CR3BP).

This module provides functions for calculating and analyzing energies and
related quantities in the CR3BP, including:
- Computing the energy (Hamiltonian) of a state
- Converting between energy and Jacobi constant
- Determining energy bounds for different regimes of motion
- Computing the potential and effective potential
"""

import math
import numpy as np
from .lagrange_points import _l1, _l2, _l3


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

def energy_to_jacobi(energy):
    """
    Convert energy to Jacobi constant.
    
    The Jacobi constant C is related to the energy E by C = -2E.
    
    Parameters
    ----------
    energy : float
        Energy value
    
    Returns
    -------
    float
        Corresponding Jacobi constant
    """
    return -2 * energy


def jacobi_to_energy(jacobi):
    """
    Convert Jacobi constant to energy.
    
    The energy E is related to the Jacobi constant C by E = -C/2.
    
    Parameters
    ----------
    jacobi : float
        Jacobi constant value
    
    Returns
    -------
    float
        Corresponding energy value
    """
    return -jacobi / 2


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

        E1 = -effective_potential_at_point(x_L1[0], 0.0, mu)  # energy level at L1
        E2 = -effective_potential_at_point(x_L2[0], 0.0, mu)  # energy level at L2
        E3 = -effective_potential_at_point(x_L3[0], 0.0, mu)  # energy level at L3
        
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


def kinetic_energy(state):
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
    return 0.5 * (vx**2 + vy**2 + vz**2)


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
    U_eff = effective_potential(state, mu)
    return 0.5 * (vx**2 + vy**2 + vz**2) + U_eff


def effective_potential(state, mu):
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
    r1 = primary_distance(state, mu)
    r2 = secondary_distance(state, mu)
    U = gravitational_potential(state, mu)
    U_eff = -0.5 * (x**2 + y**2 + z**2) + U
    return U_eff


def effective_potential_at_point(x, y, mu):
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


def gravitational_potential(state, mu):
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
    r1 = primary_distance(state, mu)
    r2 = secondary_distance(state, mu)
    U = -mu_1 / r1 - mu_2 / r2 - 0.5 * mu_1 * mu_2
    return U


def primary_distance(state, mu):
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


def secondary_distance(state, mu):
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