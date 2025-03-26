import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from src.algorithms.orbits.utils import _find_x_crossing, _halo_y, _gamma_L

def test_halo_y():
    mu = 0.01215
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.2, 0.0], dtype=np.float64)
    t1 = np.pi/2.0 - 0.15
    y_position = _halo_y(t1, 1, x0, mu)
    print("y_position:", y_position)

def test_find_x_crossing():
    mu = 0.01215
    # Some initial condition x0 for t=0
    # e.g. a typical halo orbit seed
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.2, 0.0], dtype=np.float64)

    guess_t = np.pi/2.0 - 0.15
    forward = 1  # integrate forward in time

    t_cross, x_cross = _find_x_crossing(x0, mu, forward=forward)
    print("t_cross:", t_cross)
    print("x_cross (y=0):", x_cross)

def test_gamma_L():
    mu = 0.01215
    for Lpt in [1, 2, 3]:
        gamma_value = _gamma_L(mu, Lpt)
        print(f"gamma (Lpt={Lpt}) = {gamma_value}")

if __name__ == "__main__":
    test_halo_y()
    test_find_x_crossing()
    test_gamma_L()