import numpy as np

from algorithms.dynamics.propagator import propagate_crtbp
from algorithms.orbits import HaloOrbit, LyapunovOrbit
from algorithms.manifolds.manifold import compute_manifold
from algorithms.core.lagrange_points import get_lagrange_point
from utils.plot import plot_rotating_frame_trajectories, animate_trajectories, plot_manifold
from utils.crtbp import create_3bp_system
from utils.constants import Constants
from models.body import Body


if __name__ == "__main__":

    def compute_lyapunov_manifold(mu, L_point, orbit_idx, stbl=1, direction=1, forward=1, amplitude=4e-3, use_saved=False, dx=1e-4, **ic_kwargs):
        lyapunov_orbit = LyapunovOrbit.initial_guess(mu, L_point, amplitude, **ic_kwargs)

        if use_saved:
            xL = np.load(r"src\models\xL.npy")
            t1L = np.load(r"src\models\t1L.npy")
        else:
            lyapunov_family = lyapunov_orbit.generate_family(dx=dx, forward=forward)
            xL = np.array([orbit.initial_state for orbit in lyapunov_family])
            t1L = np.array([orbit.period/2 for orbit in lyapunov_family])

        xL_i = xL[orbit_idx]
        t1L_i = t1L[orbit_idx]
        
        tf = 2 * t1L_i

        manifold_result = compute_manifold(xL_i, tf, mu, stbl, direction, forward, integration_fraction=0.7, steps=1000, tol=1e-12)
        
        return manifold_result

    def compute_halo_manifold(mu, L_point, orbit_idx, stbl=1, direction=1, forward=1, amplitude=0.2, northern=False, use_saved=False, **ic_kwargs):
        halo_orbit = HaloOrbit.initial_guess(mu, L_point, amplitude, northern, **ic_kwargs)
        halo_orbit.differential_correction()

        if use_saved:
            xH = np.load(r"src\models\xH.npy")
            t1H = np.load(r"src\models\t1H.npy")
        else:
            halo_family = halo_orbit.generate_family()
            xH = np.array([orbit.initial_state for orbit in halo_family])
            t1H = np.array([orbit.period/2 for orbit in halo_family])

        xH_i = xH[orbit_idx]
        t1H_i = t1H[orbit_idx]

        tf = 2 * t1H_i

        manifold_result = compute_manifold(xH_i, tf, mu, stbl, direction, forward, integration_fraction=0.8, steps=1000, tol=1e-12)
        
        return manifold_result

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

    mu = mu_EM

    L_point = 1
    Az = 0.2
    Ax = 4e-3

    stbl = 1
    direction = 1
    forward = -1
    
    lyapunov_manifold = compute_lyapunov_manifold(mu, L_point, 33, stbl, direction, forward, Ax, use_saved=True)
    halo_manifold = compute_halo_manifold(mu, L_point, 10, stbl, direction, forward, Az, northern=False, use_saved=True)

    xL_list = lyapunov_manifold.xW_list
    tL_list = lyapunov_manifold.tW_list

    plot_manifold([Earth, Moon], xL_list, tL_list, earth_moon_distance)

    xH_list = halo_manifold.xW_list
    tH_list = halo_manifold.tW_list
    
    plot_manifold([Earth, Moon], xH_list, tH_list, earth_moon_distance)
