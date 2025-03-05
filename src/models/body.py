"""
Celestial body model for astrodynamics simulations.

This module defines the Body class, which represents celestial bodies such as
planets, moons, or spacecraft in astrodynamics simulations. The implementation
uses Numba's jitclass for performance optimization, making it suitable for 
high-performance numerical simulations.

The Body class stores essential properties of celestial bodies including:
1. Position and velocity
2. Mass and physical radius
3. Hierarchical relationships (parent bodies)

This model supports applications in various dynamical systems, including the
Circular Restricted Three-Body Problem (CR3BP), and can be used to represent
both primary and secondary bodies, as well as test particles.
"""

from numba import types, deferred_type
from numba.experimental import jitclass

# Deferred type for circular reference
body_type = deferred_type()

spec = [
    ('name', types.unicode_type),
    ('r_init', types.float64[:]),
    ('v_init', types.float64[:]),
    ('mass', types.float64),
    ('radius', types.float64),
    ('parent', types.Optional(body_type)),
    ('parent_distance_si', types.Optional(types.float64))
]

@jitclass(spec)
class Body:
    """
    Celestial body representation for astrodynamics simulations.
    
    This class represents celestial bodies (planets, moons, spacecraft, etc.)
    with their physical properties and initial state. It is optimized for
    performance using Numba's jitclass.
    
    Parameters
    ----------
    name : str
        Name of the celestial body
    x_init : array_like
        Initial state vector [x, y, z, vx, vy, vz] in appropriate units
    mass : float
        Mass of the body
    radius : float
        Physical radius of the body
    
    Attributes
    ----------
    name : str
        Name of the celestial body
    r_init : ndarray
        Initial position vector [x, y, z]
    v_init : ndarray
        Initial velocity vector [vx, vy, vz]
    mass : float
        Mass of the body
    radius : float
        Physical radius of the body
    parent : Body, optional
        Parent body (if any) in a hierarchical system
    parent_distance_si : float, optional
        Distance to parent body in SI units (meters)
    
    Notes
    -----
    The Body class is designed to work with Numba-accelerated dynamics
    calculations. It supports hierarchical relationships between bodies,
    allowing representation of systems like star-planet-moon configurations.
    
    The initial state vector `x_init` is split into position and velocity
    components for convenience in dynamics calculations.
    """
    def __init__(self, name, x_init, mass, radius):
        """
        Initialize a celestial body with its properties and initial state.
        
        Parameters
        ----------
        name : str
            Name of the celestial body
        x_init : array_like
            Initial state vector [x, y, z, vx, vy, vz]
        mass : float
            Mass of the body
        radius : float
            Physical radius of the body
        """
        self.name = name
        self.radius = radius
        self.r_init = x_init[:3]
        self.v_init = x_init[3:]
        self.mass = mass
        self.parent = None
        self.parent_distance_si = None

# Resolve deferred type after class definition
body_type.define(Body.class_type.instance_type)

