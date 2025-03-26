"""
Stability analysis functions for the Circular Restricted Three-Body Problem (CR3BP).

This module provides functions for analyzing stability properties of equilibrium points
and periodic orbits in the CR3BP. It includes eigenvalue decomposition methods, classification
of eigenmodes, and tools for analyzing the phase space structure around libration points.
"""

import numpy as np
import warnings
import logging

from src.algorithms.dynamics.equations import jacobian_crtbp
from src.algorithms.core.lagrange_points import get_lagrange_point
from src.algorithms.manifolds.utils import _remove_infinitesimals_array, _zero_small_imag_part, _interpolate


def eigenvalue_decomposition(A, discrete=0, delta=1e-4):
    """
    Compute and classify eigenvalues and eigenvectors of a matrix into stable, 
    unstable, and center subspaces.
    
    Parameters
    ----------
    A : ndarray
        Square matrix to analyze
    discrete : int, optional
        Classification mode:
        * 0: continuous-time system (classify by real part sign)
        * 1: discrete-time system (classify by magnitude relative to 1)
    delta : float, optional
        Tolerance for classification
    
    Returns
    -------
    tuple
        (sn, un, cn, Ws, Wu, Wc) containing:
        - sn: stable eigenvalues
        - un: unstable eigenvalues
        - cn: center eigenvalues
        - Ws: eigenvectors spanning stable subspace
        - Wu: eigenvectors spanning unstable subspace
        - Wc: eigenvectors spanning center subspace
    """
    # Compute eigen-decomposition
    eigvals, eigvecs = np.linalg.eig(A)

    # Remove infinitesimal imaginary parts if an eigenvalue is "basically real"
    eigvals = np.array([_zero_small_imag_part(ev, tol=1e-14) for ev in eigvals])

    # Prepare lists
    sn, un, cn = [], [], []      # stable, unstable, center eigenvalues
    Ws_list, Wu_list, Wc_list = [], [], []  # stable, unstable, center eigenvectors

    # Classify each eigenvalue/vector, then pivot-normalize vector
    for k in range(len(eigvals)):
        val = eigvals[k]
        vec = eigvecs[:, k]

        # Find pivot (the first non-tiny entry), then normalize by that pivot
        pivot_index = 0
        while pivot_index < len(vec) and abs(vec[pivot_index]) < 1e-14:
            pivot_index += 1
        if pivot_index < len(vec):
            pivot = vec[pivot_index]
            if abs(pivot) > 1e-14:
                vec = vec / pivot

        # Optionally remove tiny real/imag parts in the vector
        vec = _remove_infinitesimals_array(vec, tol=1e-14)

        # Classification: stable/unstable/center
        if discrete == 1:
            # Discrete-time system => compare magnitude to 1 ± delta
            mag = abs(val)
            if mag < 1 - delta:
                sn.append(val)
                Ws_list.append(vec)
            elif mag > 1 + delta:
                un.append(val)
                Wu_list.append(vec)
            else:
                cn.append(val)
                Wc_list.append(vec)
        else:
            # Continuous-time system => check sign of real part
            if val.real < -delta:
                sn.append(val)
                Ws_list.append(vec)
            elif val.real > +delta:
                un.append(val)
                Wu_list.append(vec)
            else:
                cn.append(val)
                Wc_list.append(vec)

    # Convert lists into arrays
    sn = np.array(sn, dtype=np.complex128)
    un = np.array(un, dtype=np.complex128)
    cn = np.array(cn, dtype=np.complex128)

    Ws = np.column_stack(Ws_list) if Ws_list else np.zeros((A.shape[0], 0), dtype=np.complex128)
    Wu = np.column_stack(Wu_list) if Wu_list else np.zeros((A.shape[0], 0), dtype=np.complex128)
    Wc = np.column_stack(Wc_list) if Wc_list else np.zeros((A.shape[0], 0), dtype=np.complex128)

    return sn, un, cn, Ws, Wu, Wc


def libration_stability_analysis(mu, L_i, discrete=0, delta=1e-4):
    """
    Analyze the stability properties of a libration point.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system
    L_i : int
        Libration point index (1-5)
    discrete : int, optional
        Classification mode (see eigenvalue_decomposition)
    delta : float, optional
        Tolerance for classification
    
    Returns
    -------
    tuple
        (sn, un, cn, Ws, Wu, Wc) containing stability information
        (see eigenvalue_decomposition for details)
    """
    # Build the system Jacobian at L_i
    L_coords = get_lagrange_point(mu, L_i)
    A = jacobian_crtbp(L_coords[0], L_coords[1], L_coords[2], mu)

    # Compute and classify eigenvalues/vectors
    return eigenvalue_decomposition(A, discrete, delta)


def stability_indices(M):
    """
    Compute stability indices from a monodromy matrix.
    
    Parameters
    ----------
    M : array_like
        6x6 monodromy matrix from a periodic orbit
    
    Returns
    -------
    tuple
        (nu, eigvals, eigvecs) containing:
        - nu: Array of 3 stability indices
        - eigvals: Array of 6 eigenvalues of the monodromy matrix
        - eigvecs: Matrix of eigenvectors corresponding to the eigenvalues
    
    Notes
    -----
    The stability indices are computed as ν = (λ + 1/λ)/2 for each pair of
    eigenvalues. For a stable orbit, all indices must satisfy |ν| ≤ 1.
    """
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(M)
    
    # Sort eigenvalues by magnitude (descending)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Group into 3 pairs, assuming Hamiltonian dynamics where
    # eigenvalues come in reciprocal pairs (λ, 1/λ)
    nu = np.zeros(3, dtype=np.complex128)
    
    # One eigenvalue should be very close to 1 (up to numerical precision)
    # due to energy conservation
    assert np.isclose(np.abs(eigvals[0]), 1.0, rtol=1e-6), \
           "First eigenvalue should be close to 1.0"
    
    # Another eigenvalue should also be close to 1 due to periodicity
    assert np.isclose(np.abs(eigvals[1]), 1.0, rtol=1e-6), \
           "Second eigenvalue should be close to 1.0"
           
    # The remaining eigenvalues should form reciprocal pairs
    # Compute stability indices ν = (λ + 1/λ)/2
    nu[0] = (eigvals[0] + 1.0/eigvals[0]) / 2.0  # Should be ~1.0
    nu[1] = (eigvals[2] + 1.0/eigvals[2]) / 2.0
    nu[2] = (eigvals[4] + 1.0/eigvals[4]) / 2.0
    
    return nu, eigvals, eigvecs


def surface_of_section(X, T, mu, M=1, C=1):
    """
    Compute the surface-of-section for the CR3BP at specified plane crossings.
    
    This function identifies and computes the points where a trajectory crosses
    a specified plane in the phase space, creating a Poincaré section that is
    useful for analyzing the structure of the dynamics.
    
    Parameters
    ----------
    X : ndarray
        State trajectory with shape (n_points, state_dim), where each row is a
        state vector (positions and velocities), with columns representing
        [x, y, z, vx, vy, vz]
    T : ndarray
        Time stamps corresponding to the points in the state trajectory, with shape (n_points,)
    mu : float
        CR3BP mass parameter (mass ratio of smaller body to total mass)
    M : {0, 1, 2}, optional
        Determines which plane to use for the section:
        * 0: x = 0 (center-of-mass plane)
        * 1: x = -mu (larger primary plane) (default)
        * 2: x = 1-mu (smaller primary plane)
    C : {-1, 0, 1}, optional
        Crossing condition on y-coordinate:
        * 1: accept crossings with y >= 0 (default)
        * -1: accept crossings with y <= 0
        * 0: accept both y >= 0 and y <= 0
    
    Returns
    -------
    Xy0 : ndarray
        Array of state vectors at the crossing points, with shape (n_crossings, state_dim)
    Ty0 : ndarray
        Array of times corresponding to the crossing points, with shape (n_crossings,)
    
    Notes
    -----
    The function detects sign changes in the shifted x-coordinate to identify
    crossings. For M=2, it uses higher-resolution interpolation to more precisely
    locate the crossing points.
    
    Crossings are only kept if they satisfy the condition C*y >= 0, allowing
    selection of crossings in specific regions of phase space.
    """
    logger = logging.getLogger(__name__)
    
    RES = 50  # Resolution for interpolation when M=2

    try:
        # Input validation
        if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] != 6:
            logger.error(f"Invalid trajectory data: shape {X.shape if hasattr(X, 'shape') else 'unknown'}")
            return np.array([]), np.array([])
            
        if not isinstance(T, np.ndarray) or T.ndim != 1 or T.size != X.shape[0]:
            logger.error(f"Invalid time data: shape {T.shape if hasattr(T, 'shape') else 'unknown'}")
            return np.array([]), np.array([])
        
        if M not in [0, 1, 2]:
            logger.error(f"Invalid plane selector M={M}, must be 0, 1, or 2")
            return np.array([]), np.array([])
            
        if C not in [-1, 0, 1]:
            logger.error(f"Invalid crossing condition C={C}, must be -1, 0, or 1")
            return np.array([]), np.array([])

        # Determine the shift d based on M
        if M == 1:
            d = -mu
        elif M == 2:
            d = 1 - mu
        elif M == 0:
            d = 0
        
        # Copy to avoid modifying the original data
        X_copy = np.array(X, copy=True)
        T_copy = np.array(T)
        n_rows, n_cols = X_copy.shape
        
        # Shift the x-coordinate by subtracting d
        X_copy[:, 0] = X_copy[:, 0] - d
    
        # Prepare lists to hold crossing states and times
        Xy0_list = []
        Ty0_list = []
        
        if M == 1 or M == 0:
            # For M == 0 or M == 1, use the original data points
            for k in range(n_rows - 1):
                # Check if there is a sign change in the x-coordinate
                if X_copy[k, 0] * X_copy[k+1, 0] <= 0:  # Sign change or zero crossing
                    # Check the condition on y (C*y >= 0)
                    if C == 0 or np.sign(C * X_copy[k, 1]) >= 0:
                        # Choose the point with x closer to zero (to the plane)
                        K = k if abs(X_copy[k, 0]) < abs(X_copy[k+1, 0]) else k+1
                        Xy0_list.append(X[K, :])  # Use original X, not X_copy
                        Ty0_list.append(T[K])
        
        elif M == 2:
            # For M == 2, refine the crossing using interpolation
            for k in range(n_rows - 1):
                # Check if there is a sign change in the x-coordinate
                if X_copy[k, 0] * X_copy[k+1, 0] <= 0:  # Sign change or zero crossing
                    # Interpolate between the two points with increased resolution
                    dt_segment = abs(T[k+1] - T[k]) / RES
                    
                    # Make sure we have enough points for interpolation
                    if dt_segment > 0:
                        try:
                            # Use trajectory interpolation
                            XX, TT = _interpolate(X[k:k+2, :], T[k:k+2], dt_segment)
                            
                            # Also compute the shifted X values
                            XX_shifted = XX.copy()
                            XX_shifted[:, 0] = XX[:, 0] - d
                            
                            # Look through the interpolated points for the crossing
                            found_valid_crossing = False
                            for kk in range(len(TT) - 1):
                                if XX_shifted[kk, 0] * XX_shifted[kk+1, 0] <= 0:
                                    if C == 0 or np.sign(C * XX_shifted[kk, 1]) >= 0:
                                        # Choose the interpolated point closer to the plane
                                        K = kk if abs(XX_shifted[kk, 0]) < abs(XX_shifted[kk+1, 0]) else kk+1
                                        Xy0_list.append(XX[K, :])
                                        Ty0_list.append(TT[K])
                                        found_valid_crossing = True
                            
                            if not found_valid_crossing:
                                logger.debug(f"No valid crossing found after interpolation at t={T[k]:.3f}")
                        except Exception as e:
                            logger.warning(f"Interpolation failed at t={T[k]:.3f}: {str(e)}")
                            # Fallback to original point
                            K = k if abs(X_copy[k, 0]) < abs(X_copy[k+1, 0]) else k+1
                            if C == 0 or np.sign(C * X_copy[K, 1]) >= 0:
                                Xy0_list.append(X[K, :])
                                Ty0_list.append(T[K])
        
        # Convert lists to arrays
        Xy0 = np.array(Xy0_list)
        Ty0 = np.array(Ty0_list)
        
        logger.debug(f"Found {len(Xy0)} crossings for M={M}, C={C}")
        return Xy0, Ty0
    
    except Exception as e:
        logger.error(f"Error in surface_of_section: {str(e)}", exc_info=True)
        return np.array([]), np.array([]) 