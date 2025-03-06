import numpy as np

from dynamics.propagator import propagate_crtbp
from dynamics.orbits.halo import halo_orbit_ic, halo_diff_correct
from utils.plot import plot_rotating_frame_trajectories, animate_trajectories
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
    x0_guess = halo_orbit_ic(mu, Lpt, Azlp, n)
    x0_corr, half_period = halo_diff_correct(x0_guess, mu, forward=1, tol=1e-12, max_iter=250)


    tf = 2 * half_period
    sol = propagate_crtbp(x0_corr, 0.0, tf, mu)

    plot_rotating_frame_trajectories(sol, [Earth, Moon], R_earth_moon, colors=["blue", "grey"])
    animate_trajectories(sol, [Earth, Moon], R_earth_moon)
