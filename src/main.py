import numpy as np

from algorithms.dynamics.propagator import propagate_crtbp
from algorithms.orbits.halo import halo_orbit_ic, halo_family
from algorithms.manifolds.manifold import compute_manifold
from algorithms.core.lagrange_points import get_lagrange_point
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
    earth_mass = Constants.get_mass("earth")
    earth_radius = Constants.get_radius("earth")
    moon_mass = Constants.get_mass("moon")
    moon_radius = Constants.get_radius("moon")
    earth_moon_distance = Constants.get_orbital_distance("earth", "moon")

    primary_state_SJ, secondary_state_SJ, mu_SJ = create_3bp_system(sun_mass, jupiter_mass, sun_jupiter_distance)
    primary_state_EM, secondary_state_EM, mu_EM = create_3bp_system(earth_mass, moon_mass, earth_moon_distance)


    Sun = Body("Sun", primary_state_SJ, sun_mass, sun_radius)
    Jupiter = Body("Jupiter", secondary_state_SJ, jupiter_mass, jupiter_radius)
    Earth = Body("Earth", primary_state_EM, earth_mass, earth_radius)
    Moon = Body("Moon", secondary_state_EM, moon_mass, moon_radius)

    L_point = 1
    Azlp = 0.2
    n = -1
    x0_guess = halo_orbit_ic(mu_EM, L_point, Azlp, n)

    #xH, t1H = halo_family(mu_EM, L_point, x0_guess, dz=1e-4, forward=1, max_iter=250, tol=1e-12, save=True)

    xH = np.load(r"src\models\xH.npy")
    t1H = np.load(r"src\models\t1H.npy")

    idx = 10
    xH_i = xH[idx]
    t1H_i = t1H[idx]

    print(xH_i)
    print(t1H_i)

    tf = 2 * t1H_i

    stbl = 1
    direction = 1
    forward = -1

    # Get the manifold computation results
    manifold_result = compute_manifold(xH_i, tf, mu_EM, stbl, direction, forward, integration_fraction=0.8, steps=1000, tol=1e-12)
    
    # Extract the components from the result
    ysos = manifold_result.ysos
    ydsos = manifold_result.ydsos
    xH_list = manifold_result.xW_list
    tH_list = manifold_result.tW_list
    
    # Print some statistics
    print(f"Found {manifold_result.success_count} manifold crossings out of {manifold_result.attempt_count} attempts")
    print(f"Success rate: {manifold_result.success_rate:.1%}")

    plot_manifold([Sun, Jupiter], xH_list, tH_list, sun_jupiter_distance)
