import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

from src.algorithms.manifolds.analysis import eigenvalue_decomposition, surface_of_section, libration_stability_analysis
from src.algorithms.manifolds.transform import libration_to_rotating, rotating_to_libration
from src.algorithms.core.lagrange_points import get_lagrange_point
from src.algorithms.dynamics.equations import jacobian_crtbp


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
    sn, un, cn, Ws, Wu, Wc = eigenvalue_decomposition(A, discrete)

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

def test_libration_frame_eigendecomp():
    mu = 0.01215  # example Earth-Moon
    L_i = 1
    
    # We now use the libration_stability_analysis function
    sn, un, cn, Ws, Wu, Wc = libration_stability_analysis(mu, L_i)

    L_i_coords = get_lagrange_point(mu, L_i)

    A = jacobian_crtbp(L_i_coords[0], L_i_coords[1], L_i_coords[2], mu)

    sn_A, un_A, cn_A, Ws_A, Wu_A, Wc_A = eigenvalue_decomposition(A, 0)

    print(f"Stable eigenvalues OLD: {sn_A}, NEW: {sn}")
    print(f"Unstable eigenvalues OLD: {un_A}, NEW: {un}")
    print(f"Center eigenvalues OLD: {cn_A}, NEW: {cn}")

    print(f"Stable eigenvectors OLD: {Ws_A}, NEW: {Ws}")
    print(f"Unstable eigenvectors OLD: {Wu_A}, NEW: {Wu}")
    print(f"Center eigenvectors OLD: {Wc_A}, NEW: {Wc}")



if __name__ == "__main__":
    # test_surface_of_section()
    # test_eig_decomp()
    test_libration_frame_eigendecomp()
