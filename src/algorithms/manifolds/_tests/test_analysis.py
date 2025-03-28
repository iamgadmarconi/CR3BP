import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

from algorithms.manifolds.analysis import eigenvalue_decomposition, surface_of_section, libration_stability_analysis
from algorithms.manifolds.math import _libration_frame_eigenvalues, _libration_frame_eigenvectors
from algorithms.manifolds.transform import libration_to_rotating, rotating_to_libration
from algorithms.core.lagrange_points import get_lagrange_point
from algorithms.dynamics.equations import jacobian_crtbp
from models.lagrange_point import L1Point, L2Point, L3Point, L4Point, L5Point


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


def test_nonlinear_decomp():
    mu = 0.01215  # example Earth-Moon
    L_i = 1

    lin_eigenvectors = _libration_frame_eigenvectors(mu, L_i)

    _, _, _, Ws, Wu, Wc = libration_stability_analysis(mu, L_i)

    print(f"Linear eigenvectors: {lin_eigenvectors}")
    print(f"Nonlinear eigenvectors: {Ws}, {Wu}, {Wc}")


def test_comprehensive_eigendecomp():
    """
    Comprehensive test comparing linear and nonlinear eigendecomposition methods
    across various libration points and mass parameters.
    """
    print("\n=== COMPREHENSIVE EIGENDECOMPOSITION TEST ===\n")
    
    # Test different mass parameters
    mass_params = [
        (0.01215, "Earth-Moon"),      # Earth-Moon
        (0.00304, "Sun-Earth"),       # Sun-Earth
        (0.000954, "Sun-Jupiter"),    # Sun-Jupiter
        (0.1, "Heavy secondary"),     # Heavy secondary body
        (0.5, "Equal masses")         # Equal masses
    ]
    
    # Test different libration points
    for mu, system_name in mass_params:
        print(f"\n== Testing {system_name} system (μ = {mu}) ==")
        
        # For each libration point
        for L_i in range(1, 6):  # L1 through L5
            print(f"\n--- Libration Point L{L_i} ---")
            
            # Skip linearized approach for L3-L5 as it's not designed for them
            if L_i <= 2:
                # Get linearized eigenvalues and eigenvectors
                try:
                    lin_eigenvals = _libration_frame_eigenvalues(mu, L_i)
                    lin_eigenvecs = _libration_frame_eigenvectors(mu, L_i)
                    print(f"Linear eigenvalues: {lin_eigenvals}")
                    
                    # Check if the eigenvalues match the expected structure (±λ pairs)
                    assert abs(lin_eigenvals[0] + lin_eigenvals[1]) < 1e-12, "Expected ±λ pair structure"
                    assert abs(lin_eigenvals[2] + lin_eigenvals[3]) < 1e-12, "Expected ±λ pair structure"
                    
                    print("Linear eigenvalue structure verified ✓")
                except Exception as e:
                    print(f"ERROR in linear approach: {str(e)}")
            
            # Get nonlinear eigendecomposition
            try:
                sn, un, cn, Ws, Wu, Wc = libration_stability_analysis(mu, L_i)
                
                print(f"Nonlinear stable eigenvalues: {sn}")
                print(f"Nonlinear unstable eigenvalues: {un}")
                print(f"Nonlinear center eigenvalues: {cn}")
                
                # Check dimensions
                n_stable = Ws.shape[1]
                n_unstable = Wu.shape[1]
                n_center = Wc.shape[1]
                total_dim = n_stable + n_unstable + n_center
                
                print(f"Subspace dimensions - Stable: {n_stable}, Unstable: {n_unstable}, Center: {n_center}")
                assert total_dim == 6, f"Expected total dimension 6, got {total_dim}"
                
                # Verify that the eigenvectors span the correct subspaces
                L_coords = get_lagrange_point(mu, L_i)
                A = jacobian_crtbp(L_coords[0], L_coords[1], L_coords[2], mu)
                
                # Check if Ws spans stable subspace
                for i in range(n_stable):
                    v = Ws[:, i]
                    Av = A @ v
                    eigenval = sn[i]
                    err = np.linalg.norm(Av - eigenval * v) / np.linalg.norm(v)
                    assert err < 1e-10, f"Stable eigenvector {i} error: {err}"
                
                # Check if Wu spans unstable subspace
                for i in range(n_unstable):
                    v = Wu[:, i]
                    Av = A @ v
                    eigenval = un[i]
                    err = np.linalg.norm(Av - eigenval * v) / np.linalg.norm(v)
                    assert err < 1e-10, f"Unstable eigenvector {i} error: {err}"
                
                # Check if Wc spans center subspace
                for i in range(n_center):
                    v = Wc[:, i]
                    Av = A @ v
                    eigenval = cn[i]
                    err = np.linalg.norm(Av - eigenval * v) / np.linalg.norm(v)
                    assert err < 1e-10, f"Center eigenvector {i} error: {err}"
                
                print("Nonlinear eigendecomposition verified ✓")
                
                # Compare linear and nonlinear approaches for L1 and L2
                if L_i <= 2:
                    if n_unstable > 0 and len(lin_eigenvals) >= 2:
                        # Compare real eigenvalues (should correspond to unstable direction)
                        lin_real_eig = lin_eigenvals[0]
                        nonlin_real_eig = un[0] if un[0].real > 0 else sn[0]
                        rel_err = abs(lin_real_eig - nonlin_real_eig) / abs(lin_real_eig)
                        print(f"Real eigenvalue relative error: {rel_err:.6e}")
                        
                        # For collinear points, compare imaginary parts
                        if L_i <= 3:
                            lin_imag_eig = lin_eigenvals[2]
                            nonlin_imag_eig = None
                            for eig in cn:
                                if abs(eig.imag) > 1e-10:
                                    nonlin_imag_eig = eig
                                    break
                            
                            if nonlin_imag_eig is not None:
                                rel_err = abs(abs(lin_imag_eig) - abs(nonlin_imag_eig)) / abs(lin_imag_eig)
                                print(f"Imaginary eigenvalue relative error: {rel_err:.6e}")
                    
                    print("Comparison between linear and nonlinear approaches completed")
            except Exception as e:
                print(f"ERROR in nonlinear approach: {str(e)}")

def test_numerical_stability():
    """
    Test numerical stability by introducing small perturbations
    """
    print("\n=== NUMERICAL STABILITY TEST ===\n")
    
    mu = 0.01215  # Earth-Moon
    L_i = 1
    
    # Get baseline results
    sn_base, un_base, cn_base, Ws_base, Wu_base, Wc_base = libration_stability_analysis(mu, L_i)
    
    # Perturb the mass parameter slightly
    perturbations = [1e-12, 1e-10, 1e-8, 1e-6]
    
    for perturb in perturbations:
        mu_perturbed = mu * (1 + perturb)
        print(f"\nTesting with μ perturbed by {perturb}")
        
        # Get perturbed results
        sn_pert, un_pert, cn_pert, Ws_pert, Wu_pert, Wc_pert = libration_stability_analysis(mu_perturbed, L_i)
        
        # Compare eigenvalues
        stable_err = np.linalg.norm(sn_base - sn_pert) / (np.linalg.norm(sn_base) + 1e-15)
        unstable_err = np.linalg.norm(un_base - un_pert) / (np.linalg.norm(un_base) + 1e-15)
        center_err = np.linalg.norm(cn_base - cn_pert) / (np.linalg.norm(cn_base) + 1e-15)
        
        print(f"Relative changes in eigenvalues:")
        print(f"  Stable: {stable_err:.6e}")
        print(f"  Unstable: {unstable_err:.6e}")
        print(f"  Center: {center_err:.6e}")
        
        # Linear vs nonlinear stability with perturbation
        lin_eigenvals = _libration_frame_eigenvalues(mu_perturbed, L_i)
        lin_real_eig = lin_eigenvals[0]
        nonlin_real_eig = un_pert[0] if len(un_pert) > 0 else None
        
        if nonlin_real_eig is not None:
            rel_err = abs(lin_real_eig - nonlin_real_eig) / abs(lin_real_eig)
            print(f"Linear vs Nonlinear real eigenvalue relative error: {rel_err:.6e}")

def test_validate_invariant_subspaces():
    """
    Validate that the eigenvectors actually form invariant subspaces
    """
    print("\n=== INVARIANT SUBSPACE VALIDATION ===\n")
    
    mu = 0.01215  # Earth-Moon
    
    for L_i in range(1, 6):
        print(f"\n--- Testing L{L_i} ---")
        
        # Get nonlinear eigendecomposition
        sn, un, cn, Ws, Wu, Wc = libration_stability_analysis(mu, L_i)
        
        # Get system matrix
        L_coords = get_lagrange_point(mu, L_i)
        A = jacobian_crtbp(L_coords[0], L_coords[1], L_coords[2], mu)
        
        # Check if subspaces are invariant under A
        if Ws.shape[1] > 0:
            # Generate random vector in stable subspace
            v_s = np.zeros(6, dtype=complex)
            for i in range(Ws.shape[1]):
                v_s += np.random.random() * Ws[:, i]
            
            # Apply A to v_s and see if it stays in stable subspace
            Av_s = A @ v_s
            
            # Project Av_s onto Ws
            proj_s = np.zeros(6, dtype=complex)
            for i in range(Ws.shape[1]):
                proj_s += np.dot(Av_s, Ws[:, i]) * Ws[:, i]
            
            # Compute error
            err_s = np.linalg.norm(Av_s - proj_s) / np.linalg.norm(Av_s)
            print(f"Stable subspace invariance error: {err_s:.6e}")
        
        if Wu.shape[1] > 0:
            # Generate random vector in unstable subspace
            v_u = np.zeros(6, dtype=complex)
            for i in range(Wu.shape[1]):
                v_u += np.random.random() * Wu[:, i]
            
            # Apply A to v_u and see if it stays in unstable subspace
            Av_u = A @ v_u
            
            # Project Av_u onto Wu
            proj_u = np.zeros(6, dtype=complex)
            for i in range(Wu.shape[1]):
                proj_u += np.dot(Av_u, Wu[:, i]) * Wu[:, i]
            
            # Compute error
            err_u = np.linalg.norm(Av_u - proj_u) / np.linalg.norm(Av_u)
            print(f"Unstable subspace invariance error: {err_u:.6e}")
        
        if Wc.shape[1] > 0:
            # Generate random vector in center subspace
            v_c = np.zeros(6, dtype=complex)
            for i in range(Wc.shape[1]):
                v_c += np.random.random() * Wc[:, i]
            
            # Apply A to v_c and see if it stays in center subspace
            Av_c = A @ v_c
            
            # Project Av_c onto Wc
            proj_c = np.zeros(6, dtype=complex)
            for i in range(Wc.shape[1]):
                proj_c += np.dot(Av_c, Wc[:, i]) * Wc[:, i]
            
            # Compute error
            err_c = np.linalg.norm(Av_c - proj_c) / np.linalg.norm(Av_c)
            print(f"Center subspace invariance error: {err_c:.6e}")


if __name__ == "__main__":
    # test_surface_of_section()
    # test_eig_decomp()
    # test_nonlinear_decomp()
    test_comprehensive_eigendecomp()
    test_numerical_stability()
    test_validate_invariant_subspaces()
