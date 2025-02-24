"""
In the PCR3BP, find a planar Lyapunov Orbit about L1.


How do we get an initial guess for the Lyapunov Orbit?
- "Poincaré-Lindstedt Method", a high order analytic expansion
- Normal form of the Lyapunov Orbit.
- Numerical differential correction.

We want a periodic orbit \bar{x}(t) = \bar{x}(t + T)


"""
import numpy as np

from models.body import Body
from utils.crtbp import create_3bp_system, to_crtbp_units, dimless_time
from propagator import propagate_crtbp
from utils.plot import plot_trajectories

primary_state, secondary_state, mu = create_3bp_system(5.972e24, 7.348e22, 384400e3)

Earth = Body("Earth", primary_state, 5.972e24, 6378e3)
Moon = Body("Moon", secondary_state, 7.348e22, 1737e3)

Moon.parent = Earth
Moon.parent_distance_si = 384400e3

state_si = np.array([-37164e3, 0, 0,   0, -4697, 0], dtype=np.float64)

# Convert to dimensionless
state_dimless = to_crtbp_units(state_si, Earth.mass, Moon.mass, 384400e3)

days = 70

T_dimless = dimless_time(3600*24*days, Earth.mass, Moon.mass, 384400e3)

sol = propagate_crtbp(state_dimless, mu, T_dimless, steps=1000*days)

plot_trajectories(sol, [Earth, Moon], 384400e3, colors=['blue', 'grey'])
