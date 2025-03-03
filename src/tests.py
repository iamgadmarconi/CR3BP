import numpy as np

from dynamics.propagator import *
from dynamics.corrector import *
from dynamics.crtbp import *
from dynamics.orbits import *
from dynamics.manifold import *


def test_propagation_python():
    mu = 0.01215  # same as in the MATLAB example
    # Same initial condition: x, y, z, vx, vy, vz
    state0 = np.array([1.2, 0.0, 0.0, 0.0, 0.2, 0.0])
    T = 10.0

    sol = propagate_crtbp(state0, mu, T, forward=1, steps=1000)

    final_time  = sol.t[-1]
    final_state = sol.y[:, -1]

    print("Final state (Python):", final_state)
    print("Final time (Python):",  final_time)


def test_variational_equations():
    mu = 0.01215
    # 1) Same initial state
    x0_3D = np.array([1.2, 0.0, 0.0, 0.0, 0.2, 0.0], dtype=np.float64)
    # 2) Identity 6x6
    Phi0 = np.eye(6, dtype=np.float64)

    # 3) Build the initial 42-vector in *Python* layout:
    #    first 6 = state, next 36 = flatten STM
    PHI_init = np.zeros(42, dtype=np.float64)
    PHI_init[0:6] = x0_3D
    PHI_init[6:]  = Phi0.ravel()

    # 4) Integrate from t=0 to t=10
    T = 10.0
    def rhs(t, Y):
        return variational_equations(t, Y, mu)

    sol = solve_ivp(rhs, [0, T], PHI_init, rtol=1e-12, atol=1e-12, dense_output=False)

    # 5) Extract final state & STM
    PHI_final = sol.y[:, -1]
    x_final   = PHI_final[0:6]
    STM_final = PHI_final[6:].reshape((6, 6))

    print("Python variational_equations test:")
    print("Final time:", sol.t[-1])
    print("Final state:", x_final)
    print("Final STM:\n", STM_final)

if __name__ == "__main__":
    test_variational_equations()