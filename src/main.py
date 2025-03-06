import numpy as np

from dynamics.propagator import propagate_crtbp
from dynamics.orbits.halo import halo_orbit_ic
from utils.plot import plot_rotating_frame_trajectories
from utils.crtbp import create_3bp_system
from utils.constants import M_earth, M_moon, R_earth, R_moon, R_earth_moon
from models.body import Body

if __name__ == "__main__":
    primary_state, secondary_state, mu = create_3bp_system(M_earth, M_moon, R_earth_moon)
    
    Earth = Body("Earth", primary_state, M_earth, R_earth)
    Moon = Body("Moon", secondary_state, M_moon, R_moon)
    
    Lpt = 1
    Azlp = 0.2
    n = -1
    x0 = halo_orbit_ic(mu, Lpt, Azlp, n)

    tf = 2 * np.pi
    sol = propagate_crtbp(x0, 0.0, tf, mu)

    plot_rotating_frame_trajectories(sol, [Earth, Moon], R_earth_moon, colors=["blue", "grey"])
