import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dynamics.orbits.halo import halo_orbit_ic

def test_halo_orbit_ic():
    mu = 0.01215
    Lpt = 1
    Azlp = 0.2
    n = -1
    x0 = halo_orbit_ic(mu, Lpt, Azlp, n)
    print(x0)


if __name__ == "__main__":
    test_halo_orbit_ic()
