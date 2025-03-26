"""
Coordinate frame transformations for the CR3BP.

This module provides functions for transforming state vectors between different
coordinate frames used in the Circular Restricted Three-Body Problem (CR3BP):

1. Rotating frame: The standard synodic frame used in the CR3BP formulation,
   rotating with the primaries, with origin at the system barycenter.
   
2. Inertial frame: A non-rotating frame fixed in space, typically centered 
   on the primary body.
   
3. Libration frame: A local frame centered at a libration point, useful for
   analyzing motion near these equilibrium points.

These transformations are essential for analyzing trajectories in different 
reference frames, comparing CR3BP results with real-world observations, and
implementing control strategies.
"""

import numpy as np
import warnings

from src.algorithms.manifolds.math import _libration_frame_eigenvectors


def rotating_to_inertial(state_rot, t, omega, mu):
    """
    Convert state from rotating frame (CR3BP) to Earth-centered inertial frame.
    
    This function transforms a state vector from the rotating (synodic) frame
    commonly used in CR3BP to an inertial (non-rotating) frame centered on
    the primary body.
    
    Parameters
    ----------
    state_rot : array_like
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    t : float
        Time since epoch, determining the rotation angle (θ = ω*t)
    omega : float
        Angular velocity of the rotating frame (rad/time unit)
    mu : float
        Mass parameter of the CR3BP system (μ = m₂/(m₁+m₂))
        
    Returns
    -------
    ndarray
        State vector [X, Y, Z, VX, VY, VZ] in the inertial frame
        
    Notes
    -----
    The transformation includes:
    1. Shifting the origin from the barycenter to the primary body
    2. Rotating the coordinates by angle θ = ω*t
    3. Applying the velocity transformation that accounts for 
       both rotation and the Coriolis effect
    """
    r_rot = np.array(state_rot[:3])
    v_rot = np.array(state_rot[3:6])
    
    # Position relative to Earth in rotating frame
    r_rot_earth = r_rot + np.array([mu, 0, 0])
    
    theta = omega * t
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Rotation matrix (z-axis)
    R = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    
    # Rotate position to inertial frame
    r_inertial = R @ r_rot_earth
    
    # Velocity transformation: v_inertial = R(v_rot + omega × r_rot_earth)
    omega_cross_r = np.array([
        -omega * r_rot_earth[1],
        omega * r_rot_earth[0],
        0
    ])
    v_rot_earth = v_rot + omega_cross_r
    v_inertial = R @ v_rot_earth
    
    return np.concatenate([r_inertial, v_inertial])

def rotating_to_libration(state_rot, mu, L_i):
    """
    Convert state from rotating frame (CR3BP) to libration frame.
    
    This function transforms a state vector from the standard rotating frame
    to a local frame centered at one of the libration points (L1-L5).
    
    Parameters
    ----------
    state_rot : array_like
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (μ = m₂/(m₁+m₂))
    L_i : int
        Index of the libration point (1-5)
        
    Returns
    -------
    ndarray
        State vector in the libration-centered frame
        
    Notes
    -----
    The libration frame is useful for analyzing dynamics near equilibrium
    points and constructing invariant manifolds.
    """
    transform_matrix = _libration_transform_matrix(mu, L_i)
    return transform_matrix @ state_rot

def libration_to_rotating(state_lib, mu, L_i):
    """
    Convert state from libration frame to rotating frame (CR3BP).
    
    This function transforms a state vector from a local frame centered at
    a libration point back to the standard rotating frame of the CR3BP.
    
    Parameters
    ----------
    state_lib : array_like
        State vector in the libration-centered frame
    mu : float
        Mass parameter of the CR3BP system (μ = m₂/(m₁+m₂))
    L_i : ndarray
        Position vector [x, y, z] of the libration point in the rotating frame
        
    Returns
    -------
    ndarray
        State vector [x, y, z, vx, vy, vz] in the rotating frame
        
    Notes
    -----
    This is the inverse operation of rotating_to_libration().
    """
    transform_matrix = _libration_transform_matrix(mu, L_i)
    return transform_matrix.T @ state_lib

def _libration_transform_matrix(mu, L_i):
    """
    Compute the transformation matrix from rotating to libration frame.
    
    This function constructs the coordinate transformation matrix using
    the eigenvectors of the linearized dynamics near the libration point.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (μ = m₂/(m₁+m₂))
    L_i : int
        Index of the libration point (1-5)
        
    Returns
    -------
    ndarray
        6x6 transformation matrix with eigenvectors as columns
        
    Notes
    -----
    The eigenvectors define a natural basis for the dynamical flow near
    the libration point, separating the unstable, stable, and center
    subspaces.
    """
    u_1, u_2, w_1, w_2 = _libration_frame_eigenvectors(mu, L_i)
    return np.column_stack((u_1, u_2, w_1, w_2))
