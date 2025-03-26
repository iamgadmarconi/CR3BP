import numpy as np

from dynamics.propagator import propagate_crtbp
from dynamics.orbits.halo import halo_orbit_ic, halo_family
from dynamics.orbits.lyapunov import lyapunov_orbit_ic, lyapunov_family
from dynamics.manifolds.manifold import compute_manifold
from dynamics.crtbp import _l1, _l2, _l3
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

    primary_state_sj, secondary_state_sj, mu_sj = create_3bp_system(sun_mass, jupiter_mass, sun_jupiter_distance)
    primary_state_em, secondary_state_em, mu_em = create_3bp_system(earth_mass, moon_mass, earth_moon_distance)

    Sun = Body("Sun", primary_state_sj, sun_mass, sun_radius)
    Jupiter = Body("Jupiter", secondary_state_sj, jupiter_mass, jupiter_radius)
    Earth = Body("Earth", primary_state_em, earth_mass, earth_radius)
    Moon = Body("Moon", secondary_state_em, moon_mass, moon_radius)

    L_point = 1
    Azlp = 0.2
    n = -1
    x0_guess_halo = halo_orbit_ic(mu_em, L_point, Azlp, n)
    x0_guess_lyapunov = lyapunov_orbit_ic(mu_em, L_point, Ax=4e-3)

    # xH, t1H = halo_family(mu_em, L_point, x0_guess_halo, dz=1e-4, forward=1, max_iter=250, tol=1e-12, save=True)
    print(x0_guess_lyapunov)
    xL, t1L = lyapunov_family(mu_em, L_point, x0_guess_lyapunov, forward=1, max_iter=250, tol=1e-12, save=True)

    xH = np.load(r"src\models\xH.npy")
    t1H = np.load(r"src\models\t1H.npy")
    xL = np.load(r"src\models\xL.npy")
    t1L = np.load(r"src\models\t1L.npy")


    idx = 31
    xH_i = xH[idx]
    t1H_i = t1H[idx]

    xL_i = xL[idx]
    t1L_i = t1L[idx]

    print(xH_i)
    print(t1H_i)

    print(xL_i)
    print(t1L_i)

    tf_halo = 2 * t1H_i
    tf_lyapunov = 2 * t1L_i

    stbl = 1
    direction = 1
    forward = -1

    _, _, xH_list, tH_list = compute_manifold(xH_i, tf_halo, mu_em, stbl, direction, forward, integration_fraction=0.8, steps=1000, tol=1e-12)
    _, _, xL_list, tL_list = compute_manifold(xL_i, tf_lyapunov, mu_em, stbl, direction, forward, integration_fraction=0.7, steps=1000, tol=1e-12)

    plot_manifold([Earth, Moon], xH_list, tH_list, earth_moon_distance)
    plot_manifold([Earth, Moon], xL_list, tL_list, earth_moon_distance)
