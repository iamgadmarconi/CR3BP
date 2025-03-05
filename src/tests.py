import numpy as np

from dynamics.propagator import *
from dynamics.stm import *
from dynamics.stm import _compute_stm
from dynamics.dynamics import *
from dynamics.crtbp import *
from dynamics.crtbp import _l1
from dynamics.orbits.lyapunov import *
from dynamics.orbits.utils import *
from dynamics.manifolds.math import *
from dynamics.manifolds.manifold import *
from dynamics.manifolds.manifold import _compute_manifold_section
from dynamics.manifolds.utils import *

from utils.plot import *


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

def test_halo_y():
    mu = 0.01215
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.2, 0.0], dtype=np.float64)
    t1 = np.pi/2.0 - 0.15
    y_position = halo_y(t1, 1, x0, mu)
    print("y_position:", y_position)

def test_find_x_crossing():
    mu = 0.01215
    # Some initial condition x0 for t=0
    # e.g. a typical halo orbit seed
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.2, 0.0], dtype=np.float64)

    guess_t = np.pi/2.0 - 0.15
    forward = 1  # integrate forward in time

    t_cross, x_cross = find_x_crossing(x0, mu, forward=forward)
    print("t_cross:", t_cross)
    print("x_cross (y=0):", x_cross)


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
    mu = 0.012150585609624
    L_i = _l1(mu)
    
    # Example seed guess for planar orbit:
    # x0, y0, z0, vx0, vy0, vz0
    # For purely planar you might set z0=0, vz0=0:
    x0i = np.array([0.840895693043321, 0.0, 0.0, 0.0, -0.0334899524017813, 0.0])

    # Generate the family
    xL, t1L = lyapunov_family(mu, L_i, x0i, forward=1,
                              max_iter=250, tol=1e-12, save=True)
    print("xL shape:", xL.shape)
    print("t1L shape:", t1L.shape)


def test_surface_of_section():
    # 1) Build a sample trajectory that crosses x=0
    t_span = np.arange(0, 10.0, 0.1)
    x_traj = 0.05 * np.sin(2.0 * np.pi * 0.2 * t_span)  # crosses zero
    y_traj = 0.10 * np.cos(2.0 * np.pi * 0.1 * t_span)
    vx_traj = np.zeros_like(t_span)
    vy_traj = np.zeros_like(t_span)
    
    # Combine into single X array
    # The columns are assumed [x, y, vx, vy, ...] in your usage
    X = np.column_stack((x_traj, y_traj, vx_traj, vy_traj))
    
    # 2) Call the surface_of_section function
    mu = 0.01215  # example Earth-Moon
    M = 1         # same setting as in the MATLAB test
    C = 1         # y >= 0
    Xy0, Ty0 = surface_of_section(X, t_span, mu, M=M, C=C)
    
    # 3) Print results & do a quick check
    print(f"Number of crossings: {Xy0.shape[0]}")
    print("First few crossing states:")
    print(Xy0[:5, :])
    
    print("Corresponding crossing times:")
    print(Ty0[:5])
    
    # If you want a quick check that x-d is near zero:
    # For M=1 => d=-mu => x - (-mu) = x+mu => should be ~0 at crossing
    # print("Check that x+mu is near zero:")
    # print(Xy0[:5, 0] + mu)

    # 4) Optional: plot the trajectory and crossing points
    plt.figure()
    plt.plot(x_traj, y_traj, 'k-', label='Trajectory')
    plt.plot(Xy0[:,0], Xy0[:,1], 'ro', label='SoS Crossings')
    plt.title('Surface-of-section test')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
    print("test_surface_of_section completed successfully.")


def test_eig_decomp():
    # 1) Build a sample matrix A, same as the MATLAB example
    # We'll use a diagonal for clarity: [0.9, 1.1, 1.0, 1.0]
    A = np.array([[ 5,  3,  5],
                [ -3,  5,  5],
                [ 2,   -3,  2]])
    # 2) Call the eig_decomp function
    discrete = 1
    sn, un, cn, Ws, Wu, Wc = eig_decomp(A, discrete)

    # 3) Print the results
    print("Stable eigenvalues:", sn)
    print("Unstable eigenvalues:", un)
    print("Center eigenvalues:", cn)

    print("Stable eigenvectors:", Ws)
    print("Unstable eigenvectors:", Wu)
    print("Center eigenvectors:", Wc)

    print("Stable subspace dimension:", Ws.shape[1])
    print("Unstable subspace dimension:", Wu.shape[1])
    print("Center subspace dimension:", Wc.shape[1])

    # 4) Optional: verify that A * w_s ~ sn(i) * w_s, etc.
    # For stable eigenvectors:
    for i in range(Ws.shape[1]):
        test_vec = Ws[:,i]
        check_resid = A @ test_vec - sn[i]*test_vec
        print(f"Ws vector {i} residue norm:", np.linalg.norm(check_resid))

    print("test_eig_decomp completed successfully.")

def test_compute_manifold_section():
    # Example 6D initial condition on a 3D orbit
    x0 = np.array([0.843995693043320, 0, 0, 0, -0.0565838306397683, 0])
    T = 2.70081224387894
    frac = 0.98
    stbl = 1        # stable manifold
    direction = 1   # positive branch
    mu = 0.01215     # Earth-Moon ratio (example)
    NN = 1


    xx, tt, phi_T, PHI = _compute_stm(x0, mu, T, forward=1)
    print(f"phi_T: {phi_T}")

    x0W = _compute_manifold_section(x0, T, frac, stbl, direction, mu, NN, forward=1)
    # print("Manifold initial state x0W:", x0W)

if __name__ == "__main__":
    # test_propagation_python()
    # test_variational_equations()
    # test_compute_stm()
    # test_haloy()cccccccccccc
    # test_find_x_crossing()
    # test_lyapunov_diff_correct()
    # test_lyapunov_family()
    # t1L = np.load(r'src\models\t1L.npy')
    # xL = np.load(r'src\models\xL.npy')
    # print(t1L)
    # print(xL[330])
    # test_surface_of_section()
    # test_eig_decomp()
    test_compute_manifold_section()