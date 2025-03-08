import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

from src.dynamics.orbits.halo import halo_orbit_ic, halo_diff_correct

def test_halo_orbit_ic():
    mu = 0.0121505856
    Lpt = 1
    Azlp = 0.2
    n = -1
    x0 = halo_orbit_ic(mu, Lpt, Azlp, n)
    print(x0)

def test_halo_diff_correct():
    mu = 0.0121505856  # Example CR3BP parameter (Earth–Moon, etc.)
    # State vector: [x, y, z, vx, vy, vz]
    x0_guess = np.array([0.823451685541845, 0, 0.032462441320139, 0, 0.142149195738938, 0])

    # Run the Python differential corrector routine
    x0_corr, half_period = halo_diff_correct(x0_guess, mu, tol=1e-12, max_iter=250)

    # Compare the outputs using a tolerance for floating-point errors.
    print("Computed x0_corr:", x0_corr)
    print("Computed half period:", half_period)


if __name__ == "__main__":
    test_halo_orbit_ic()
    test_halo_diff_correct()
