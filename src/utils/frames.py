import numpy as np

def rotating_to_inertial(state_rot, t, omega, mu):
    """
    Convert state from rotating frame (CR3BP) to Earth-centered inertial frame.
    
    Args:
        state_rot: Array-like [x, y, z, vx, vy, vz] in rotating frame
        t: Time since epoch (rotation angle = omega * t)
        omega: Angular velocity of rotating frame (rad/time unit)
        mu: CR3BP mass parameter (dimensionless)
        
    Returns:
        state_inertial: Numpy array [X, Y, Z, VX, VY, VZ] in Earth-centered inertial frame
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
