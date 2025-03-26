"""
Coordinate transformations for the Circular Restricted Three-Body Problem (CR3BP).

This module provides functions for transforming coordinates between different reference
frames used in the CR3BP, particularly for libration point analysis. It includes conversions
between rotating frame, libration-centered frames, and normalized coordinates used in 
stability analysis.
"""

import numpy as np

from src.algorithms.core.lagrange_points import lagrange_point_locations, get_lagrange_point


def libration_to_rotating(state, mu, L_i):
    """
    Transform coordinates from libration point frame to rotating frame.
    
    Parameters
    ----------
    state : array_like
        State vector [x, y, z, vx, vy, vz] in libration point frame
    mu : float
        Mass parameter of the CR3BP system
    L_i : int
        Libration point index (1-5)
    
    Returns
    -------
    ndarray
        State vector in rotating frame
    """
    # Get libration point coordinates
    L_pos = get_lagrange_point(mu, L_i)
    
    # Initialize output state with same shape as input
    rotating_state = np.zeros_like(state)
    
    # Transform position components
    rotating_state[0] = state[0] + L_pos[0]
    rotating_state[1] = state[1] + L_pos[1]
    rotating_state[2] = state[2] + L_pos[2]
    
    # Transform velocity components (no change needed for simple translation)
    rotating_state[3:] = state[3:]
    
    return rotating_state


def rotating_to_libration(state, mu, L_i):
    """
    Transform coordinates from rotating frame to libration point frame.
    
    Parameters
    ----------
    state : array_like
        State vector [x, y, z, vx, vy, vz] in rotating frame
    mu : float
        Mass parameter of the CR3BP system
    L_i : int
        Libration point index (1-5)
    
    Returns
    -------
    ndarray
        State vector in libration point frame
    """
    # Get libration point coordinates
    L_pos = get_lagrange_point(mu, L_i)
    
    # Initialize output state with same shape as input
    libration_state = np.zeros_like(state)
    
    # Transform position components
    libration_state[0] = state[0] - L_pos[0]
    libration_state[1] = state[1] - L_pos[1]
    libration_state[2] = state[2] - L_pos[2]
    
    # Transform velocity components (no change needed for simple translation)
    libration_state[3:] = state[3:]
    
    return libration_state


def primary_barycentric_distance(mu, primary_index):
    """
    Compute the distance from the barycenter to a primary body.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system
    primary_index : int
        Primary body index (1 for larger primary, 2 for smaller primary)
    
    Returns
    -------
    float
        Distance from the barycenter to the specified primary body
    """
    if primary_index == 1:
        return -mu
    elif primary_index == 2:
        return 1 - mu
    else:
        raise ValueError("Primary index must be 1 or 2") 