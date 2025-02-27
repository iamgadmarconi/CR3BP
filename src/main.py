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
from dynamics.propagator import propagate_crtbp
from dynamics.crtbp import compute_energy_bounds, _energy_to_jacobi_constant, _l1, _l2
from dynamics.orbits import general_linear_ic, lyapunov_orbit_ic, lyapunov_family
from dynamics.corrector import lyapunov_diff_correct
from utils.plot import (plot_rotating_frame_trajectories, 
                        plot_inertial_frame_trajectories, 
                        animate_trajectories,
                        plot_libration_points,
                        plot_zvc)
from utils.frames import libration_to_rotating, _mu_bar, _libration_frame_eigenvectors



if __name__ == "__main__":

    show_plots = True

    primary_state, secondary_state, mu = create_3bp_system(5.972e24, 7.348e22, 384400e3)

    # print(primary_state, secondary_state, mu)

    Earth = Body("Earth", primary_state, 5.972e24, 6378e3)
    Moon = Body("Moon", secondary_state, 7.348e22, 1737e3)

    Moon.parent = Earth
    Moon.parent_distance_si = 384400e3

    l1 = _l1(mu)
    l2 = _l2(mu)

    # print(l1, l2)

    T = 2 * np.pi

    t_final = T

    # state0 = [l1[0] - 1e-3, 0, 0, 0, 0, 0]

    # sol = propagate_crtbp(state0, mu, t_final, 1000)

    # if show_plots:
    #     plot_rotating_frame_trajectories(sol, [Earth, Moon], 384400e3, colors=['blue', 'grey'])
    #     plot_inertial_frame_trajectories(sol, [Earth, Moon], 384400e3, colors=['blue', 'grey'])
    #     animate_trajectories(sol, [Earth, Moon], 384400e3)

    # state0 = [l1[0] + 1e-3, 0, 0, 0, 0, 0]

    # sol = propagate_crtbp(state0, mu, t_final, 1000)

    # if show_plots:
    #     plot_rotating_frame_trajectories(sol, [Earth, Moon], 384400e3, colors=['blue', 'grey'])
    #     plot_inertial_frame_trajectories(sol, [Earth, Moon], 384400e3, colors=['blue', 'grey'])
    #     animate_trajectories(sol, [Earth, Moon], 384400e3)

    mu_bar = _mu_bar(mu, l1)
    # print(mu_bar)

    u1, u2, u, v = _libration_frame_eigenvectors(mu, l1, orbit_type="lyapunov")
    # print(f'u1: {u1}, u2: {u2}, u: {u}, v: {v}')

    initial_guess = lyapunov_orbit_ic(mu, l1, 1e-3)
    #print(f'initial_guess: {initial_guess}')

    # sol = propagate_crtbp(initial_guess, mu, t_final, 1000)

    # if show_plots:
    #     plot_rotating_frame_trajectories(sol, [Earth, Moon], 384400e3, colors=['blue', 'grey'])
    #     plot_inertial_frame_trajectories(sol, [Earth, Moon], 384400e3, colors=['blue', 'grey'])
    #     animate_trajectories(sol, [Earth, Moon], 384400e3)

    x0_corrected, half_period = lyapunov_diff_correct(initial_guess, mu, tol=1e-12, max_iter=25)

    print(f'Initial guess: {initial_guess}, Corrected: {x0_corrected}, half_period: {half_period}')
    # sol = propagate_crtbp(x0_corrected, mu, half_period*16, 1000)

    # if show_plots:
    #     plot_rotating_frame_trajectories(sol, [Earth, Moon], 384400e3, colors=['blue', 'grey'])
    #     plot_inertial_frame_trajectories(sol, [Earth, Moon], 384400e3, colors=['blue', 'grey'])
    #     animate_trajectories(sol, [Earth, Moon], 384400e3)

    xL, t1L = lyapunov_family(mu, l1, x0_corrected)
    print(f'xL: {xL}, t1L: {t1L}')
