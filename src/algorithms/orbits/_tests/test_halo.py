import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

from src.algorithms.orbits.halo import halo_orbit_ic, halo_diff_correct
from src.algorithms.orbits import HaloOrbit

def test_halo_orbit_ic():
    mu = 0.0121505856
    Lpt = 1
    Azlp = 0.2
    # Test with northern=False (equivalent to n=-1 in old function)
    x0 = halo_orbit_ic(mu, Lpt, Az=Azlp, northern=False)
    print(f"Initial condition: {x0}")
    print(f"Position: {x0[:3]}")
    print(f"Velocity: {x0[3:]}")

def test_halo_diff_correct():
    mu = 0.0121505856  # Example CR3BP parameter (Earth–Moon, etc.)
    # State vector: [x, y, z, vx, vy, vz]
    x0_guess = np.array([0.823451685541845, 0, 0.032462441320139, 0, 0.142149195738938, 0])

    # Run the Python differential corrector routine
    x0_corr, half_period = halo_diff_correct(x0_guess, mu, tol=1e-12, max_iter=250)

    # Compare the outputs using a tolerance for floating-point errors.
    print("Computed x0_corr:", x0_corr)
    print("Computed half period:", half_period)

def test_halo_orbit_class():
    """Test the new HaloOrbit class"""
    mu = 0.0121505856  # Earth-Moon system
    L_i = 1
    
    # Test initial_guess class method
    orbit = HaloOrbit.initial_guess(mu, L_i, amplitude=0.02, northern=False)
    print(f"Initial guess state: {orbit.initial_state}")
    
    # Test differential correction method
    orbit.differential_correction(tol=1e-12, max_iter=250)
    print(f"Corrected state: {orbit.initial_state}")
    print(f"Half period: {orbit.period/2}")
    
    # Test propagation
    times, trajectory = orbit.propagate(steps=200)
    print(f"Trajectory shape: {trajectory.shape}")
    
    # Test stability computation
    stability_info = orbit.compute_stability()
    print(f"Stability indices: {stability_info[0]}")
    print(f"Is stable: {orbit.is_stable}")

def test_halo_family_generation():
    """Test generating a family of halo orbits using the OO approach"""
    mu = 0.0121505856  # Earth-Moon system
    L_i = 1
    
    # Create an initial orbit
    orbit = HaloOrbit.initial_guess(mu, L_i, amplitude=0.01)
    orbit.differential_correction()
    
    # Generate a small family of orbits
    z_range = np.linspace(0.01, 0.03, 5)
    family = orbit.generate_family(z_range, save=False)
    
    print(f"Generated {len(family)} orbits")
    print(f"Z amplitudes: {[orb.initial_state[2] for orb in family]}")
    
    # Test energy and Jacobi constant computation
    energies = [orb.energy for orb in family]
    print(f"Energy range: [{energies[0]}, {energies[-1]}]")


if __name__ == "__main__":
    test_halo_orbit_ic()
    test_halo_diff_correct()
    print("\nTesting HaloOrbit class:")
    test_halo_orbit_class()
    print("\nTesting HaloOrbit family generation:")
    test_halo_family_generation()
