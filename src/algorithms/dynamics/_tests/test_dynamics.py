import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
# Use a direct import after fixing the path
from algorithms.dynamics.propagator import propagate_variational_equations


def test_variational_equations():
    # 1) Define mu and forward
    mu = 0.01215   # Earth-Moon approximate
    forward = 1    # +1 for forward in time

    # 2) Build initial 6x6 STM (identity) and state
    Phi0 = np.eye(6, dtype=np.float64)
    x0_3D = np.array([1.2, 0.0, 0.0, 0.0, 0.2, 0.0], dtype=np.float64)

    # 3) Pack into 42-vector in *MATLAB layout*
    PHI_init = np.zeros(42, dtype=np.float64)
    PHI_init[:36] = Phi0.ravel()  # Flattened 6x6
    PHI_init[36:] = x0_3D         # [x, y, z, vx, vy, vz]

    # 4) Integrate for T=10
    T = 10.0
    sol = propagate_variational_equations(PHI_init, mu, T, forward=forward)

    # 5) Extract final STM + state
    final_phivec = sol.y[:, -1]  # shape (42,)
    STM_final = final_phivec[:36].reshape((6, 6))
    state_final = final_phivec[36:]

    print("Final time:", sol.t[-1])
    print("Final state:", state_final)
    print("Final STM:\n", STM_final)

if __name__ == "__main__":
    test_variational_equations()