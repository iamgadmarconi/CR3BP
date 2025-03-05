import numba
import numpy as np

from src.utils.constants import G

@numba.njit(fastmath=True, cache=True)
def create_3bp_system(primary_mass, secondary_mass, distance):
    """
    Create a 3BP system with primary and secondary bodies.

    Parameters
    ----------
    primary_mass : float
        Mass of primary body in kilograms.
    secondary_mass : float
        Mass of secondary body in kilograms.
    distance : float
        Distance between primary and secondary bodies in meters.

    Returns
    ------- 
    primary_state : np.ndarray[6,]
        State of primary body in dimensionless units.
    secondary_state : np.ndarray[6,]
        State of secondary body in dimensionless units.
    mu : float
        Mass parameter (mu = m2/(m1 + m2)).
    """
    # Mass parameter (mu = m2/(m1 + m2))
    mu = _get_mass_parameter(primary_mass, secondary_mass)

    # Barycentric positions
    primary_pos_x = -mu * distance
    secondary_pos_x = (1 - mu) * distance

    # Angular velocity calculation (ω)
    omega = _get_angular_velocity(primary_mass, secondary_mass, distance)

    # Barycentric velocities (y-direction for circular orbit)
    primary_vel_y = -omega * abs(primary_pos_x)  # Earth moves downward
    secondary_vel_y = omega * secondary_pos_x          # Moon moves upward

    primary_state_si = np.array([primary_pos_x, 0.0, 0.0, 0.0, primary_vel_y, 0.0], dtype=np.float64)
    secondary_state_si = np.array([secondary_pos_x, 0.0, 0.0, 0.0, secondary_vel_y, 0.0], dtype=np.float64)

    primary_state = to_crtbp_units(primary_state_si, primary_mass, secondary_mass, distance)
    secondary_state = to_crtbp_units(secondary_state_si, primary_mass, secondary_mass, distance)

    return primary_state, secondary_state, mu

@numba.njit(fastmath=True, cache=True)
def _get_mass_parameter(primary_mass, secondary_mass):
    return secondary_mass / (primary_mass + secondary_mass)

@numba.njit(fastmath=True, cache=True)
def _get_angular_velocity(primary_mass, secondary_mass, distance):
    return np.sqrt(G * (primary_mass + secondary_mass) / distance**3)

@numba.njit(fastmath=True, cache=True)
def to_crtbp_units(state_si, m1, m2, distance):
    """
    Convert an SI-state vector into the dimensionless state used by crtbp_accel.
    
    Parameters
    ----------
    state_si  : array-like of shape (6,)
        [x, y, z, vx, vy, vz] in meters and meters/sec, all in Earth-centered coordinates
        if you're modeling Earth-Moon, for example.
    m1        : float
        Mass of primary (e.g., Earth) in kilograms.
    m2        : float
        Mass of secondary (e.g., Moon) in kilograms.
    distance  : float
        Distance between the two main bodies (e.g., Earth-Moon) in meters.
        
    Returns
    -------
    state_dimless : np.ndarray of shape (6,)
        The dimensionless state vector suitable for crtbp_accel.
    mu            : float
        Dimensionless mass parameter = m2 / (m1 + m2).
    """
    # Mean motion (rad/s) => in CRTBP, we want n = 1, so we scale by this factor.
    n = _get_angular_velocity(m1, m2, distance)

    # Compute the dimensionless mass parameter
    mu = _get_mass_parameter(m1, m2)

    # Position scaled by the chosen distance
    x_star = state_si[0] / distance
    y_star = state_si[1] / distance
    z_star = state_si[2] / distance

    # Velocity scaled by distance * n
    vx_star = state_si[3] / (distance * n)
    vy_star = state_si[4] / (distance * n)
    vz_star = state_si[5] / (distance * n)

    state_dimless = np.array([x_star, y_star, z_star, vx_star, vy_star, vz_star], dtype=np.float64)
    return state_dimless

@numba.njit(fastmath=True, cache=True)
def to_si_units(state_dimless, m1, m2, distance):
    """
    Convert a dimensionless state vector into the SI-state vector used by crtbp_accel.

    Parameters
    ----------
    state_dimless : np.ndarray of shape (6,)
        The dimensionless state vector suitable for crtbp_accel.
    m1        : float
        Mass of primary (e.g., Earth) in kilograms.
    m2        : float
        Mass of secondary (e.g., Moon) in kilograms.
    distance  : float
        Distance between the two main bodies (e.g., Earth-Moon) in meters.

    Returns
    -------
    state_si : np.ndarray of shape (6,)
        The SI-state vector suitable for crtbp_accel.
    """
    n = _get_angular_velocity(m1, m2, distance)

    x = state_dimless[0] * distance
    y = state_dimless[1] * distance
    z = state_dimless[2] * distance

    vx = state_dimless[3] * distance * n
    vy = state_dimless[4] * distance * n
    vz = state_dimless[5] * distance * n

    return np.array([x, y, z, vx, vy, vz], dtype=np.float64)

@numba.njit(fastmath=True, cache=True)
def dimless_time(T, m1, m2, distance):
    n = _get_angular_velocity(m1, m2, distance)
    return T * n

@numba.njit(fastmath=True, cache=True)
def si_time(T_dimless, m1, m2, distance):
    n = _get_angular_velocity(m1, m2, distance)
    return T_dimless / n

@numba.njit(fastmath=True, cache=True)
def get_distance(state_1_nondim, state_0_nondim, system_distance):
    """
    Calculate physical distance between two bodies in meters.
    
    Parameters
    ----------
    state_1_nondim : np.ndarray[6,]
        First body's dimensionless state vector
    state_0_nondim : np.ndarray[6,]
        Second body's dimensionless state vector  
    system_distance : float
        Actual distance between primary bodies in meters (conversion factor)
        
    Returns
    -------
    float
        Physical distance between bodies in meters
    """
    # Get position components (first 3 elements) from dimensionless states
    pos_diff = state_1_nondim[:3] - state_0_nondim[:3]
    
    # Calculate dimensionless distance (normalized by system_distance)
    dimless_dist = np.linalg.norm(pos_diff)
    
    # Convert to physical distance in meters
    return dimless_dist * system_distance
