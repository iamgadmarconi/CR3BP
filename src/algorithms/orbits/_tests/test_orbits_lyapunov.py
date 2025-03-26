import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from src.algorithms.orbits.lyapunov import lyapunov_diff_correct, lyapunov_family, lyapunov_orbit_ic
from src.algorithms.core.lagrange_points import get_lagrange_point


def test_lyapunov_orbit_ic():
    mu = 0.0121505856  # Example CR3BP parameter (Earth–Moon, etc.)
    L_i = 1
    x0i = lyapunov_orbit_ic(mu, L_i, Ax=1e-5)
    print("x0i shape:", x0i.shape)

def test_lyapunov_diff_correct():
    mu = 0.0121505856  # Example CR3BP parameter (Earth–Moon, etc.)
    # State vector: [x, y, z, vx, vy, vz]
    x0_guess = np.array([0.840895693043321, 0.0, 0.0, 0.0, -0.0334899524017813, 0.0])

    # Run the Python differential corrector routine
    x0_corr, half_period = lyapunov_diff_correct(x0_guess, mu, forward=1, tol=1e-12, max_iter=50)

    # Compare the outputs using a tolerance for floating-point errors.
    print("Computed x0_corr:", x0_corr)
    print("Computed half period:", half_period)

def test_lyapunov_family():
    # Earth-Moon system parameters
    mu = 0.0121505856
    L_i = 1

    x0i = np.array([0.840895693043321, 0.0, 0.0, 0.0, -0.0334899524017813, 0.0])

    # Generate the family
    xL, t1L = lyapunov_family(mu, L_i, x0i, forward=1,
                                max_iter=250, tol=1e-12, save=False)
    print("xL shape:", xL.shape)
    print("t1L shape:", t1L.shape)


if __name__ == "__main__":
    test_lyapunov_orbit_ic()
    test_lyapunov_diff_correct()
    test_lyapunov_family()
