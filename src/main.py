import numpy as np
import matplotlib.pyplot as plt
from models.body import Body
from utils.crtbp import create_3bp_system, to_crtbp_units, dimless_time
from dynamics.propagator import propagate_crtbp
from dynamics.crtbp import compute_energy_bounds, _energy_to_jacobi_constant, _l1, _l2
from dynamics.orbits import general_linear_ic, lyapunov_orbit_ic, lyapunov_family
from dynamics.corrector import lyapunov_diff_correct, compute_stm
from dynamics.manifold import compute_manifold
from utils.plot import (plot_rotating_frame_trajectories, 
                        plot_inertial_frame_trajectories, 
                        animate_trajectories,
                        plot_libration_points,
                        plot_zvc,
                        plot_orbit_family,
                        plot_orbit_family_energy,
                        plot_manifold)
from utils.frames import libration_to_rotating, _mu_bar, _libration_frame_eigenvectors



if __name__ == "__main__":

    show_plots = True

    primary_state, secondary_state, mu = create_3bp_system(5.972e24, 7.348e22, 384400e3)
    Earth = Body("Earth", primary_state, 5.972e24, 6378e3)
    Moon = Body("Moon", secondary_state, 7.348e22, 1737e3)
    Moon.parent = Earth
    Moon.parent_distance_si = 384400e3

    l1 = _l1(mu)
    l2 = _l2(mu)

    T = 2 * np.pi
    t_final = T
    mu_bar = _mu_bar(mu, l1)

    u1, u2, u, v = _libration_frame_eigenvectors(mu, l1, orbit_type="lyapunov")

    initial_guess = lyapunov_orbit_ic(mu, l1, 4e-3)

    x0_corrected, half_period = lyapunov_diff_correct(initial_guess, mu, tol=1e-12, max_iter=25)

    # print(f'Initial guess: {initial_guess}, Corrected: {x0_corrected}, half_period: {half_period}')

    # xL, t1L = lyapunov_family(mu, l1, initial_guess, save=True)
    # print(f'xL: {xL}, t1L: {t1L}')

    xL = np.load(r'src\models\xL.npy')
    t1L = np.load(r'src\models\t1L.npy')
    np.set_printoptions(threshold=np.inf)
    # print(f'xL: {xL}')

    idx = 31
    x0 = xL[idx]
    T = t1L[idx]
    stbl = 1
    direction = 1

    ysos, ydsos, xW_list, tW_list = compute_manifold(x0, T, mu, stbl, direction, step=0.02)

    plot_manifold([Earth, Moon], xW_list, tW_list, 384400e3)
