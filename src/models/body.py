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
    def __init__(self, name, x_init, mass, radius):
        self.name = name
        self.radius = radius
        self.r_init = x_init[:3]
        self.v_init = x_init[3:]
        self.mass = mass
        self.parent = None
        self.parent_distance_si = None

# Resolve deferred type after class definition
body_type.define(Body.class_type.instance_type)

