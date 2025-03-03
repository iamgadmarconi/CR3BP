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
    sol = integrate_variational_equations(PHI_init, mu, T, forward=forward,
                                        rtol=1e-12, atol=1e-12, steps=500)

    # 5) Extract final STM + state
    final_phivec = sol.y[:, -1]  # shape (42,)
    STM_final = final_phivec[:36].reshape((6, 6))
    state_final = final_phivec[36:]

    print("Final time:", sol.t[-1])
    print("Final state:", state_final)
    print("Final STM:\n", STM_final)

def test_compute_stm():
    mu = 0.01215
    x0 = np.array([1.2, 0.0, 0.0, 0.0, 0.2, 0.0], dtype=np.float64)
    tf = 6.28   # approximate "one revolution" time

    # Forward integration
    x_fwd, t_fwd, phiT_fwd, PHI_fwd = compute_stm(x0, mu, tf, forward=-1)
    print("Forward integration result:")
    print("Times:", t_fwd)
    print("Final time:", t_fwd[-1])
    print("Final state:", x_fwd[-1])
    print("Monodromy matrix:\n", phiT_fwd)
    print("Final row of PHI (STM + state):\n", PHI_fwd[-1])

if __name__ == "__main__":
    test_compute_stm()
