import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from src.dynamics.stm import _compute_stm

def test_compute_stm_output():
    # Define a sample 6-element initial condition, mass ratio, and final time
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    mu = 0.0121505856    # Example mass ratio (e.g., Earth-Moon system)
    tf = 2 * np.pi       # One period (adjust as needed)
    forward = 1

    # Call the Python function
    x, t, phi_T, PHI = _compute_stm(x0, mu, tf, forward=forward)

    # Print outputs for manual comparison
    print("=== Python _compute_stm Output ===")
    print("Time vector (first 5 and last 5 values):")
    print("t[0:5] =", t[:5])
    print("t[-5:] =", t[-5:])
    print("\nFinal state at tf (x[-1]):")
    print(x[-1])
    print("\nMonodromy matrix at tf (phi_T):")
    print(phi_T)
    print("\nFull integrated solution shape (PHI):", PHI.shape)

def test_with_known_solution():
    x0 = np.array([0.843995693043320, 0, 0, 0, -0.0565838306397683, 0])
    T = 2.70081224387894
    forward = 1
    mu = 0.0121505856

    x, t, phi_T, PHI = _compute_stm(x0, mu, T, forward=forward)

    # Print outputs for manual comparison
    print("=== Python _compute_stm With Known Solution Output ===")
    print("Time vector (first 5 and last 5 values):")
    print("t[0:5] =", t[:5])
    print("t[-5:] =", t[-5:])
    print("\nFinal state at tf (x[-1]):")
    print(x[-1])
    print("\nMonodromy matrix at tf (phi_T):")
    print(phi_T)
    print("\nFull integrated solution shape (PHI):", PHI.shape)

if __name__ == "__main__":
    test_compute_stm_output()
    test_with_known_solution()
