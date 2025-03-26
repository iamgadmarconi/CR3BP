"""
Stability analysis functions for the Circular Restricted Three-Body Problem (CR3BP).

This module provides functions for analyzing stability properties of equilibrium points
and periodic orbits in the CR3BP. It includes eigenvalue decomposition methods, classification
of eigenmodes, and tools for analyzing the phase space structure around libration points.
"""

import numpy as np
import warnings

from src.algorithms.dynamics.equations import jacobian_crtbp
from src.algorithms.core.lagrange_points import get_lagrange_point
from src.algorithms.manifolds.utils import _remove_infinitesimals_array, _zero_small_imag_part


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
    Compute the intersection of a trajectory with a Poincaré section.
    
    Parameters
    ----------
    X : ndarray
        State trajectory, shape (n_points, 6)
    T : ndarray
        Time vector, shape (n_points,)
    mu : float
        Mass parameter of the CR3BP system
    M : int, optional
        Primary body index to define the section plane (1 or 2)
    C : int, optional
        Type of constraint:
        * 1: section with y=0, vy>0 (default)
        * 2: section with y=0, vy<0
        * 3: section with x=const
    
    Returns
    -------
    tuple
        (Xy0, Ty0) containing:
        - Xy0: States at section crossings
        - Ty0: Times at section crossings
    """
    if M not in [1, 2]:
        raise ValueError("M must be 1 (primary) or 2 (secondary)")
    
    if C not in [1, 2, 3]:
        raise ValueError("C must be 1, 2, or 3")
    
    # Extract coordinates
    x = X[:, 0]
    y = X[:, 1]
    vx = X[:, 3]
    vy = X[:, 4]
    
    # Primary locations
    xm1 = -mu
    xm2 = 1 - mu
    
    # Find indices where sign changes occur
    if C == 1 or C == 2:
        # Find indices where y changes sign
        inds = np.where(y[:-1] * y[1:] <= 0)[0]
        
        # Apply additional velocity constraint
        if len(inds) > 0:
            if C == 1:  # vy > 0
                inds = [i for i in inds if (vy[i] + vy[i+1])/2 > 0]
            else:       # vy < 0
                inds = [i for i in inds if (vy[i] + vy[i+1])/2 < 0]
    
    elif C == 3:
        # Get the x value for the specified primary
        xm = xm1 if M == 1 else xm2
        
        # Find indices where x - xm changes sign
        x_rel = x - xm
        inds = np.where(x_rel[:-1] * x_rel[1:] <= 0)[0]
    
    if len(inds) == 0:
        return np.array([]), np.array([])
    
    # Interpolate to find exact crossing points
    Xy0 = []
    Ty0 = []
    
    from src.algorithms.manifolds.utils import _interpolate
    
    for i in inds:
        # Get the two points that bracket the crossing
        X1 = X[i]
        X2 = X[i+1]
        t1 = T[i]
        t2 = T[i+1]
        
        # Determine interpolation parameter
        if C == 1 or C == 2:
            # Interpolate based on y=0
            s = -X1[1] / (X2[1] - X1[1])
        else:
            # Interpolate based on x=xm
            xm = xm1 if M == 1 else xm2
            s = (xm - X1[0]) / (X2[0] - X1[0])
        
        # Ensure s is in [0, 1]
        s = max(0, min(1, s))
        
        # Interpolate state and time
        Xs = _interpolate(X1, X2, s)
        ts = (1-s)*t1 + s*t2
        
        Xy0.append(Xs)
        Ty0.append(ts)
    
    return np.array(Xy0), np.array(Ty0) 