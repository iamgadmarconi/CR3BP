"""
Physical constants for astrodynamics simulations.

This module contains fundamental physical constants and system-specific values
for use in astrodynamics simulations. All constants are defined in SI units
and stored as numpy float64 data types for precision and consistency in 
numerical computations.

The module includes:
1. Universal physical constants (gravitational constant)
2. Planetary and lunar masses
3. Characteristic distances for common systems

These constants provide the foundation for various dynamical calculations
including the Circular Restricted Three-Body Problem (CR3BP) and orbital
mechanics problems.

References
----------
Values are based on standard IAU (International Astronomical Union) and
NASA/JPL data. For detailed sources, see:
- IAU 2015 Resolution B3 (https://www.iau.org/static/resolutions/IAU2015_English.pdf)
- NASA JPL Solar System Dynamics (https://ssd.jpl.nasa.gov/)
"""

import numpy as np

# Universal physical constants
#-----------------------------

#: float: Universal gravitational constant (m^3 kg^-1 s^-2)
#: Defines the strength of gravitational interaction
G = np.float64(6.67430e-11)  # m^3 kg^-1 s^-2

# Celestial body masses
#---------------------

#: float: Mass of Earth (kg)
#: Standard mass of Earth, the primary body in Earth-centered systems
M_earth = np.float64(5.972e24)  # kg

#: float: Mass of Moon (kg)
#: Standard mass of the Moon, Earth's natural satellite
M_moon = np.float64(7.348e22)  # kg

#: float: Mass of Sun (kg)
#: Standard mass of the Sun, the central body of the solar system
M_sun = np.float64(1.989e30)  # kg

# Characteristic distances
#------------------------

#: float: Average Earth-Moon distance (m)
#: Semi-major axis of the Moon's orbit around Earth
R_earth_moon = np.float64(384400e3)  # m

#: float: Average Earth-Sun distance (m)
#: Semi-major axis of the Earth's orbit around the Sun
R_earth_sun = np.float64(149.6e9)  # m



# Body radii
#-----------

#: float: Radius of Earth (m)
R_earth = np.float64(6378.137e3)  # m

#: float: Radius of Moon (m)
R_moon = np.float64(1737.4e3)  # m

#: float: Radius of Sun (m)
R_sun = np.float64(696340e3)  # m






