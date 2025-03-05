import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from src.dynamics.manifolds.manifold import _compute_manifold_section, compute_manifold


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

if __name__ == "__main__":
    test_compute_manifold_section()
