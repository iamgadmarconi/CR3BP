import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from src.dynamics.manifolds.manifold import _compute_manifold_section, compute_manifold
from src.utils.plot import plot_manifold
from src.utils.crtbp import create_3bp_system
from src.models.body import Body

def test_compute_manifold_section():
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


def test_compute_manifold():
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

if __name__ == "__main__":
    test_compute_manifold_section()
    test_compute_manifold()
