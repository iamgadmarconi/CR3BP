import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from src.dynamics.manifolds.manifold import _compute_manifold_section, compute_manifold
from src.dynamics.orbits.halo import halo_orbit_ic, halo_diff_correct
from src.dynamics.propagator import propagate_crtbp
from src.utils.plot import plot_manifold, plot_rotating_frame_trajectories
from src.utils.crtbp import create_3bp_system
from src.utils.constants import M_earth, M_moon, R_earth_moon, R_earth, R_moon
from src.models.body import Body



def test_compute_manifold_section_lyapunov():
    # Example 6D initial condition on a 3D orbit
    x0 = np.array([0.843995693043320, 0, 0, 0, -0.0565838306397683, 0])
    T = 2.70081224387894
    frac = 0.98
    stbl = 1        # stable manifold
    direction = 1   # positive branch
    mu = 0.0121505856     # Earth-Moon ratio (example)
    NN = 1

    x0W = _compute_manifold_section(x0, T, frac, stbl, direction, mu, NN, forward=1)
    print("Manifold initial state x0W:", x0W)


def test_compute_manifold_lyapunov():
    x0g = np.array([0.840895693043321, 0, 0, 0, -0.033489952401781, 0])
    T = 2.70081224387894
    mu = 0.0121505856

    forward = -1
    step = 0.02

    xL = np.load(r'src\models\xL.npy')
    t1L = np.load(r'src\models\t1L.npy')

    idx = 32
    x0 = xL[idx]
    T = t1L[idx]
    stbl = 1
    direction = 1

    ysos, ydsos, xW_list, tW_list = compute_manifold(x0, 2*T, mu, stbl=stbl, direction=direction, forward=forward, step=step, rtol=3e-14, atol=1e-14)

    primary_state, secondary_state, mu = create_3bp_system(5.972e24, 7.348e22, 384400e3)
    Earth = Body("Earth", primary_state, 5.972e24, 6378e3)
    Moon = Body("Moon", secondary_state, 7.348e22, 1737e3)
    plot_manifold([Earth, Moon], xW_list, tW_list, 384400e3)


def test_compute_manifold_section_halo():
    primary_state, secondary_state, mu = create_3bp_system(M_earth, M_moon, R_earth_moon)
    
    Earth = Body("Earth", primary_state, M_earth, R_earth)
    Moon = Body("Moon", secondary_state, M_moon, R_moon)
    
    Lpt = 1
    Azlp = 0.2
    n = -1
    x0_guess = halo_orbit_ic(mu, Lpt, Azlp, n)
    x0_corr, half_period = halo_diff_correct(x0_guess, mu, forward=1, tol=1e-12, max_iter=250)


    tf = 2 * half_period

    frac = 0.8
    stbl = 1
    forward = -1
    direction = 1

    x0W = _compute_manifold_section(x0_corr, tf, frac, stbl, direction, mu, 1, forward=forward)
    print("Manifold initial state x0W:", x0W)

    T = 2 * np.pi * 0.78
    sol = propagate_crtbp(x0W, 0.0, T, mu, forward=forward)
    plot_rotating_frame_trajectories(sol, [Earth, Moon], R_earth_moon, colors=["blue", "grey"])


def test_compute_manifold_halo():
    primary_state, secondary_state, mu = create_3bp_system(M_earth, M_moon, R_earth_moon)
    
    Earth = Body("Earth", primary_state, M_earth, R_earth)
    Moon = Body("Moon", secondary_state, M_moon, R_moon)
    
    Lpt = 3
    Azlp = 0.3
    n = -1

    x0_guess = halo_orbit_ic(mu, Lpt, Azlp, n)

    stbl = 1
    forward = -1
    direction = 1
    step = 0.01

    x0_corr, half_period = halo_diff_correct(x0_guess, mu, tol=3e-14, max_iter=250)

    tf = 2 * half_period

    ysos, ydsos, xW_list, tW_list = compute_manifold(x0_corr, 2*tf, mu, stbl=stbl,
                                    direction=direction, forward=forward, step=step,
                                    integration_fraction=0.8, rtol=3e-14, atol=1e-14
                                    )
    plot_manifold([Earth, Moon], xW_list, tW_list, R_earth_moon)

if __name__ == "__main__":
    # test_compute_manifold_section_lyapunov()
    # test_compute_manifold_lyapunov()
    # test_compute_manifold_section_halo()
    test_compute_manifold_halo()
