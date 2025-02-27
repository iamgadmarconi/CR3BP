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

def rotating_to_libration(state_rot, t, omega, mu, L_i):
    """
    Convert state from rotating frame (CR3BP) to libration frame.
    """
    transform_matrix = _libration_transform_matrix(mu, L_i)
    return transform_matrix @ state_rot

def _libration_transform_matrix(mu, L_i):
    u_1, u_2, w_1, w_2 = _libration_frame_eigenvectors(mu, L_i)
    return np.column_stack((u_1, u_2, w_1, w_2))

def _libration_frame_eigenvalues(mu_bar, L_i):
    """
    Compute the eigenvalues of the libration frame.

    Args:
        mu: CR3BP mass parameter (dimensionless)
        L_i: Libration point coordinates in dimensionless units

    Returns:
        eigenvalues: Numpy array of eigenvalues
    """
    alpha_1 = _alpha_1(mu_bar, L_i)
    alpha_2 = _alpha_2(mu_bar, L_i)

    lambda_1 = np.sqrt(alpha_1)
    lambda_2 = - np.sqrt(alpha_1)
    v_1 = np.sqrt(- alpha_2)
    v_2 = - np.sqrt(-alpha_2)

    return lambda_1, lambda_2, v_1, v_2

def _libration_frame_eigenvectors(mu, L_i):
    """
    Compute the eigenvectors of the libration frame.
    """

    mu_bar = _mu_bar(mu, L_i)
    lambda_1, lambda_2, v_1, v_2 = _libration_frame_eigenvalues(mu_bar, L_i)

    a = 2 * mu_bar + 1
    b = mu_bar - 1

    sigma = 2 * lambda_1 / (lambda_1**2 + b)
    tau = - (v_1 **2 + a) / (2*v_1)

    u_1 = np.array([1, -sigma, lambda_1, lambda_2*sigma])
    u_2 = np.array([1, sigma, lambda_2, lambda_2*sigma])
    w_1 = np.array([1, -1j*tau, 1j*v_1, v_1*tau])
    w_2 = np.array([1, 1j*tau, 1j*v_2, v_1*tau])

    return u_1, u_2, w_1, w_2

def _mu_bar(mu, L_i):
    """
    Compute the reduced mass parameter.
    """
    x_L_i = L_i[0]
    return mu * np.abs(x_L_i - 1 + mu) ** (-3) + (1 - mu) * np.abs(x_L_i + mu) ** (-3)

def _alpha_1(mu, L_i):
    """
    Compute the first eigenvalue of the libration frame.
    """
    mu_bar = _mu_bar(mu, L_i)
    alpha = (mu_bar - 2 + np.emath.sqrt(9*mu_bar**2 - 8*mu_bar)) / 2
    if isinstance(alpha, np.complex128):
        raise ValueError("Alpha 1 is complex")
    return alpha

def _alpha_2(mu, L_i):
    """
    Compute the second eigenvalue of the libration frame.
    """
    mu_bar = _mu_bar(mu, L_i)
    alpha = (mu_bar - 2 - np.emath.sqrt(9*mu_bar**2 - 8*mu_bar)) / 2
    if isinstance(alpha, np.complex128):
        raise ValueError("Alpha 2 is complex")
    return alpha

def _beta_1(mu, L_i):
    """
    Compute the first beta coefficient of the libration frame.
    """
    beta = np.emath.sqrt(_alpha_1(mu, L_i))
    if isinstance(beta, np.complex128):
        raise ValueError("Beta 1 is complex")
    return beta

def _beta_2(mu, L_i):
    """
    Compute the second beta coefficient of the libration frame.
    """
    beta = np.emath.sqrt(_alpha_2(mu, L_i))
    if isinstance(beta, np.complex128):
        raise ValueError("Beta 2 is complex")
    return beta

