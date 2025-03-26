import numpy as np

from algorithms.dynamics.propagator import propagate_crtbp
from algorithms.orbits.halo import halo_orbit_ic, halo_family
from algorithms.manifolds.manifold import compute_manifold
from algorithms.dynamics.crtbp import _l1, _l2, _l3
from utils.plot import plot_rotating_frame_trajectories, animate_trajectories, plot_manifold
from utils.crtbp import create_3bp_system
from utils.constants import Constants
from models.body import Body

if __name__ == "__main__":

    sun_mass = Constants.get_mass("sun")
    sun_radius = Constants.get_radius("sun")
    jupiter_mass = Constants.get_mass("jupiter")
    jupiter_radius = Constants.get_radius("jupiter")
    sun_jupiter_distance = Constants.get_orbital_distance("sun", "jupiter")

    primary_state, secondary_state, mu = create_3bp_system(sun_mass, jupiter_mass, sun_jupiter_distance)
    
    Sun = Body("Sun", primary_state, sun_mass, sun_radius)
    Jupiter = Body("Jupiter", secondary_state, jupiter_mass, jupiter_radius)

    L_point = 2
    Azlp = 0.2
    n = -1
    x0_guess = halo_orbit_ic(mu, L_point, Azlp, n)

    # xH, t1H = halo_family(mu, L_point, x0_guess, dz=1e-4, forward=1, max_iter=250, tol=1e-12, save=True)

    xH = np.load(r"src\models\xH.npy")
    t1H = np.load(r"src\models\t1H.npy")

    idx = 39
    xH_i = xH[idx]
    t1H_i = t1H[idx]

    print(xH_i)
    print(t1H_i)

    tf = 2 * t1H_i

    stbl = 1
    direction = 1
    forward = -1

    ysos, ydsos, xH_list, tH_list = compute_manifold(xH_i, tf, mu, stbl, direction, forward, integration_fraction=0.8, steps=1000, tol=1e-12)

    plot_manifold([Sun, Jupiter], xH_list, tH_list, sun_jupiter_distance)
