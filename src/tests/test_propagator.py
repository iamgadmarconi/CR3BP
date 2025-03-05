import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from src.dynamics.propagator import propagate_crtbp, propagate_variational_equations

def test_propagation_python():
    mu = 0.01215  # same as in the MATLAB example
    # Same initial condition: x, y, z, vx, vy, vz
    state0 = np.array([1.2, 0.0, 0.0, 0.0, 0.2, 0.0])
    T = 10.0

    sol = propagate_crtbp(state0, 0, T, mu, forward=1, steps=1000)

    final_time  = sol.t[-1]
    final_state = sol.y[:, -1]

    print("Final state (Python):", final_state)
    print("Final time (Python):",  final_time)


def test_propagation_variational_equations():
    mu = 0.01215  # same as in the MATLAB example
    # Same initial condition: x, y, z, vx, vy, vz
    phi_init = np.zeros(42) # 42 element vector
    x0 = np.array([1.2, 0.0, 0.0, 0.0, 0.2, 0.0])
    phi_init[:36] = np.eye(6).flatten() # 36 elements
    phi_init[36:] = x0 # 6 elements
    T = 10.0
    forward = 1
    sol = propagate_variational_equations(phi_init, mu, T, forward=forward)

    final_time  = sol.t[-1]
    final_state = sol.y[:, -1]

    print("Final state (Python):", final_state)
    print("Final time (Python):",  final_time)

if __name__ == "__main__":
    test_propagation_python()
    test_propagation_variational_equations()