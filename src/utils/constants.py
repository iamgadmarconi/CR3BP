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

#: float: Mass of Sun (kg)
#: Standard mass of the Sun, the central body of the solar system
M_sun = np.float64(1.989e30)  # kg

#: float: Mass of Mercury (kg)
#: Standard mass of Mercury, the smallest planet in the solar system
M_mercury = np.float64(3.302e23)  # kg

#: float: Mass of Venus (kg)
#: Standard mass of Venus, the second planet from the Sun
M_venus = np.float64(4.867e24)  # kg

#: float: Mass of Earth (kg)
#: Standard mass of Earth, the primary body in Earth-centered systems
M_earth = np.float64(5.972e24)  # kg

#: float: Mass of Moon (kg)
#: Standard mass of the Moon, Earth's natural satellite
M_moon = np.float64(7.348e22)  # kg

#: float: Mass of Mars (kg)
#: Standard mass of Mars, the fourth planet from the Sun
M_mars = np.float64(6.417e23)  # kg

#: float: Mass of Phobos (kg)
#: Standard mass of Phobos, the larger moon of Mars
M_phobos = np.float64(1.072e16)  # kg

#: float: Mass of Deimos (kg)
#: Standard mass of Deimos, the smaller moon of Mars
M_deimos = np.float64(1.476e15)  # kg

#: float: Mass of Jupiter (kg)
#: Standard mass of Jupiter, the largest planet in the solar system
M_jupiter = np.float64(1.898e27)  # kg

#: float: Mass of Io (kg)
#: Standard mass of Io, the innermost moon of Jupiter
M_io = np.float64(8.932e22)  # kg

#: float: Mass of Europa (kg)
#: Standard mass of Europa, the second-largest moon of Jupiter
M_europa = np.float64(4.8e22)  # kg

#: float: Mass of Ganymede (kg)
#: Standard mass of Ganymede, the largest moon of Jupiter
M_ganymede = np.float64(1.482e23)  # kg

#: float: Mass of Callisto (kg)
#: Standard mass of Callisto, the third-largest moon of Jupiter
M_callisto = np.float64(1.076e23)  # kg

#: float: Mass of Saturn (kg)
#: Standard mass of Saturn, the sixth planet from the Sun
M_saturn = np.float64(5.683e26)  # kg

#: float: Mass of Titan (kg)
#: Standard mass of Titan, the largest moon of Saturn
M_titan = np.float64(1.345e23)  # kg

#: float: Mass of Uranus (kg)
#: Standard mass of Uranus, the seventh planet from the Sun
M_uranus = np.float64(8.681e25)  # kg

#: float: Mass of Neptune (kg)
#: Standard mass of Neptune, the eighth planet from the Sun
M_neptune = np.float64(1.024e26)  # kg

#: float: Mass of Triton (kg)
#: Standard mass of Triton, the largest moon of Neptune
M_triton = np.float64(2.14e22)  # kg

# Characteristic distances
#------------------------

# Average Sun-Mercury distance (m)
R_sun_mercury = np.float64(57.91e9)  # m

#: float: Average Sun-Venus distance (m)
#: Semi-major axis of the Venus's orbit around Sun
R_sun_venus = np.float64(108.2e9)  # m

#: float: Average Earth-Sun distance (m)
#: Semi-major axis of the Earth's orbit around the Sun
R_earth_sun = np.float64(149.6e9)  # m

#: float: Average Earth-Moon distance (m)
#: Semi-major axis of the Moon's orbit around Earth
R_earth_moon = np.float64(384400e3)  # m


#: float: Average Sun-Mars distance (m)
#: Semi-major axis of the Mars's orbit around Sun
R_sun_mars = np.float64(227.9e9)  # m

#: float: Average Mars-Phobos distance (m)
#: Semi-major axis of the Phobos's orbit around Mars
R_mars_phobos = np.float64(9248e3)  # m

#: float: Average Mars-Deimos distance (m)
#: Semi-major axis of the Deimos's orbit around Mars
R_mars_deimos = np.float64(23460e3)  # m

#: float: Average Sun-Jupiter distance (m)
#: Semi-major axis of the Jupiter's orbit around Sun
R_sun_jupiter = np.float64(778.5e9)  # m

#: float: Average Jupiter-Io distance (m)
#: Semi-major axis of the Io's orbit around Jupiter
R_jupiter_io = np.float64(421700e3)  # m

#: float: Average Jupiter-Europa distance (m)
#: Semi-major axis of the Europa's orbit around Jupiter
R_jupiter_europa = np.float64(671100e3)  # m

#: float: Average Jupiter-Ganymede distance (m)
#: Semi-major axis of the Ganymede's orbit around Jupiter
R_jupiter_ganymede = np.float64(1070400e3)  # m

#: float: Average Jupiter-Callisto distance (m)
#: Semi-major axis of the Callisto's orbit around Jupiter
R_jupiter_callisto = np.float64(1882700e3)  # m

#: float: Average Sun-Saturn distance (m)
#: Semi-major axis of the Saturn's orbit around Sun
R_sun_saturn = np.float64(1426.7e9)  # m

#: float: Average Saturn-Titan distance (m)
#: Semi-major axis of the Titan's orbit around Saturn
R_saturn_titan = np.float64(1221870e3)  # m

#: float: Average Sun-Uranus distance (m)
#: Semi-major axis of the Uranus's orbit around Sun
R_sun_uranus = np.float64(2870.97e9)  # m

#: float: Average Sun-Neptune distance (m)
#: Semi-major axis of the Neptune's orbit around Sun
R_sun_neptune = np.float64(4498.25e9)  # m

#: float: Average Neptune-Triton distance (m)
#: Semi-major axis of the Triton's orbit around Neptune
R_neptune_triton = np.float64(354759e3)  # m

#: float: Average Sun-Pluto distance (m)
#: Semi-major axis of the Pluto's orbit around Sun
R_sun_pluto = np.float64(5906.38e9)  # m

# Body radii
#-----------

#: float: Radius of Sun (m)
R_sun = np.float64(696340e3)  # m

#: float: Radius of Mercury (m)
R_mercury = np.float64(2439.7e3)  # m

#: float: Radius of Venus (m)
R_venus = np.float64(6051.8e3)  # m

#: float: Radius of Earth (m)
R_earth = np.float64(6378.137e3)  # m

#: float: Radius of Moon (m)
R_moon = np.float64(1737.4e3)  # m

#: float: Radius of Mars (m)
R_mars = np.float64(3396.2e3)  # m

#: float: Radius of Phobos (m)
R_phobos = np.float64(11.269e3)  # m

#: float: Radius of Deimos (m)
R_deimos = np.float64(6.2e3)  # m

#: float: Radius of Jupiter (m)
R_jupiter = np.float64(69911e3)  # m

#: float: Radius of Io (m)
R_io = np.float64(1821.6e3)  # m

#: float: Radius of Europa (m)
R_europa = np.float64(1560.8e3)  # m

#: float: Radius of Ganymede (m)
R_ganymede = np.float64(2631.2e3)  # m

#: float: Radius of Callisto (m)
R_callisto = np.float64(2410.3e3)  # m

#: float: Radius of Saturn (m)
R_saturn = np.float64(58232e3)  # m

#: float: Radius of Titan (m)
R_titan = np.float64(2574.7e3)  # m

#: float: Radius of Uranus (m)
R_uranus = np.float64(25362e3)  # m

#: float: Radius of Neptune (m)
R_neptune = np.float64(24622e3)  # m

#: float: Radius of Triton (m)
R_triton = np.float64(1737.4e3)  # m

#: float: Radius of Pluto (m)
R_pluto = np.float64(1188.3e3)  # m







