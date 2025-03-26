"""
Computation of Lagrange (libration) points in the CR3BP.

This module provides functions for calculating the positions of the five
Lagrange points in the Circular Restricted Three-Body Problem (CR3BP).
"""

import numpy as np
import mpmath as mp
import warnings

# Set mpmath precision to 50 digits for root finding
mp.mp.dps = 50


def lagrange_point_locations(mu):
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


def get_lagrange_point(mu, point_index):
    """
    Get the position of a specific Lagrange point.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    point_index : int
        Lagrange point index (1-5)
        
    Returns
    -------
    ndarray
        3D vector [x, y, z] giving the position of the specified Lagrange point
    """
    if point_index == 1:
        return _l1(mu)
    elif point_index == 2:
        return _l2(mu)
    elif point_index == 3:
        return _l3(mu)
    elif point_index == 4:
        return _l4(mu)
    elif point_index == 5:
        return _l5(mu)
    else:
        raise ValueError("Invalid Lagrange point index. Must be 1-5.")


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
    return x - (1 - mu) * (x + mu) / (r1**3) - mu * (x - (1 - mu)) / (r2**3) 