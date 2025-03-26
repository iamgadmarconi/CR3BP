import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

from src.dynamics.manifolds.math import _surface_of_section, _eig_decomp, _libration_frame_eigendecomp, _libration_frame_eigenvalues, _libration_frame_eigenvectors
from src.dynamics.crtbp import _libration_index_to_coordinates
from src.dynamics.dynamics import jacobian_crtbp


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
    Xy0, Ty0 = _surface_of_section(X, t_span, mu, M=M, C=C)
    
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
    sn, un, cn, Ws, Wu, Wc = _eig_decomp(A, discrete)

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
    eig1, eig2, eig3, eigv1, eigv2, eigv3 = _libration_frame_eigendecomp(mu, L_i)

    L_i_coords = _libration_index_to_coordinates(mu, L_i)

    A = jacobian_crtbp(L_i_coords[0], L_i_coords[1], L_i_coords[2], mu)

    eig1_A, eig2_A, eig3_A, eigV1_A, eigV2_A, eigV3_A = _eig_decomp(A, 0)

    print(f"First eigenvalue OLD: {eig1_A}, NEW: {eig1}")
    print(f"Second eigenvalue OLD: {eig2_A}, NEW: {eig2}")
    print(f"Third eigenvalue OLD: {eig3_A}, NEW: {eig3}")

    print(f"First eigenvector OLD: {eigV1_A}, NEW: {eigv1}")
    print(f"Second eigenvector OLD: {eigV2_A}, NEW: {eigv2}")
    print(f"Third eigenvector OLD: {eigV3_A}, NEW: {eigv3}")



if __name__ == "__main__":
    # test_surface_of_section()
    # test_eig_decomp()
    test_libration_frame_eigendecomp()
