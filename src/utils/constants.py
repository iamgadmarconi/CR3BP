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

# Characteristic distances
#------------------------

#: float: Average Earth-Moon distance (m)
#: Semi-major axis of the Moon's orbit around Earth
R_earth_moon = np.float64(384400e3)  # m





