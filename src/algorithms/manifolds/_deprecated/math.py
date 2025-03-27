"""
Mathematical functions for analyzing manifolds in the Circular Restricted Three-Body Problem (CR3BP).

This module provides mathematical utilities for analyzing the dynamical structure
of the CR3BP, particularly related to libration points, eigenvalue decomposition,
and Poincaré sections. It includes functions for computing eigenvalues and eigenvectors
of the linearized system around libration points, as well as tools for finding
surface-of-section crossings in trajectory data.

These functions form the mathematical foundation for computing and analyzing
stable and unstable manifolds in the CR3BP.
"""

import numpy as np
import warnings

from src.algorithms.manifolds.utils import _remove_infinitesimals_array, _zero_small_imag_part, _interpolate
from src.algorithms.core.lagrange_points import get_lagrange_point
from src.algorithms.dynamics.equations import jacobian_crtbp


def _libration_frame_eigendecomp(mu, L_i, discrete=0, delta=1e-4):
    """
    Compute and classify the eigenvalues of the linearized system around a 
    libration point, sorting them into stable (sn), unstable (un), and 
    center (cn) subspaces, just like _eig_decomp.

    Parameters
    ----------
    mu : float
        CR3BP mass parameter (mass ratio of smaller body to total mass)
    L_i : int
        Libration point index (1-5)
    discrete : int, optional
        1 = discrete-time classification, 0 = continuous-time classification
    delta : float, optional
        Tolerance used in classifying near-unit-magnitude eigenvalues 
        (if discrete=1) or near-zero real part (if discrete=0)

    Returns
    -------
    sn : np.ndarray
        Stable eigenvalues (continuous-time: real part < 0)
    un : np.ndarray
        Unstable eigenvalues (continuous-time: real part > 0)
    cn : np.ndarray
        Center eigenvalues (continuous-time: real part = 0)
    Ws : np.ndarray
        Eigenvectors spanning stable subspace
    Wu : np.ndarray
        Eigenvectors spanning unstable subspace
    Wc : np.ndarray
        Eigenvectors spanning center subspace
    """
    # Build the system Jacobian at L_i
    L_coords = get_lagrange_point(mu, L_i)
    A = jacobian_crtbp(L_coords[0], L_coords[1], L_coords[2], mu)

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

def _libration_frame_eigenvalues(mu, L_i):
    """
    Compute the eigenvalues of the linearized system around a libration point.
    
    This function calculates the four eigenvalues of the linearized dynamics 
    around a libration point in the CR3BP, which characterize the local stability
    properties of the point.
    
    Parameters
    ----------
    mu : float
        CR3BP mass parameter (mass ratio of smaller body to total mass)
    L_i : int
        Index of the libration point (1-5)
    
    Returns
    -------
    tuple
        A tuple of four eigenvalues (lambda1, -lambda1, lambda2, -lambda2)
        representing the characteristic dynamics around the libration point
    
    Notes
    -----
    The eigenvalues come in pairs due to the Hamiltonian structure of the system.
    Typically, for collinear libration points (L1, L2, L3), there will be one
    real pair (±λ) corresponding to the hyperbolic (unstable) direction, and
    one imaginary pair (±iν) corresponding to the elliptic (center) direction.
    """
    alpha_1 = _alpha_1(mu, L_i)
    alpha_2 = _alpha_2(mu, L_i)

    eig1 = np.sqrt(alpha_1)
    eig2 = np.emath.sqrt(-alpha_2)

    return eig1, -eig1, eig2, -eig2

def _libration_frame_eigenvectors(mu, L_i, orbit_type="lyapunov"):
    """
    Compute the eigenvectors of the linearized system around a libration point.
    
    This function calculates the eigenvectors corresponding to the eigenvalues of
    the linearized dynamics around a libration point. These eigenvectors define
    the principal directions of motion near the libration point.
    
    Parameters
    ----------
    mu : float
        CR3BP mass parameter (mass ratio of smaller body to total mass)
    L_i : int
        Index of the libration point (1-5)
    orbit_type : str, optional
        Type of orbit to consider:
        * "lyapunov": planar Lyapunov orbits (default)
        * any other value: three-dimensional orbits (e.g., "halo", "vertical")
    
    Returns
    -------
    tuple
        A tuple of eigenvectors corresponding to different modes of motion:
        * For orbit_type="lyapunov": (u_1, u_2, u, v)
        * For other orbit types: (u_1, u_2, w_1, w_2)
        
        where:
        - u_1, u_2 are the eigenvectors associated with the real eigenvalues
        - w_1, w_2 are the eigenvectors associated with the imaginary eigenvalues
        - u, v are special vectors used for Lyapunov orbit computation
    
    Notes
    -----
    The eigenvectors provide the fundamental directions that define the phase space
    structure around the libration point, including the stable, unstable, and
    center manifolds that organize the dynamics in the vicinity of the libration point.
    """
    lambda_1, lambda_2, nu_1, nu_2 = _libration_frame_eigenvalues(mu, L_i)

    a = _a(mu, L_i)
    b = _b(mu, L_i)

    sigma = _sigma(mu, L_i)
    tau = _tau(mu, L_i)

    u_1 = np.array([1, -sigma, lambda_1, lambda_2*sigma])
    u_2 = np.array([1, sigma, lambda_2, lambda_2*sigma])
    w_1 = np.array([1, -1j*tau, 1j*nu_1, nu_1*tau])
    w_2 = np.array([1, 1j*tau, 1j*nu_2, nu_1*tau])
    u = np.array([1, 0, 0, nu_1 * tau])
    v = np.array([0, tau, nu_2, 0])

    if orbit_type == "lyapunov":
        return u_1, u_2, u, v
    else:
        return u_1, u_2, w_1, w_2

def _mu_bar(mu, L_i):
    """
    Compute the reduced mass parameter for a libration point.
    
    This function calculates the reduced mass parameter μ̄, which is a key parameter
    in the linearized dynamics around a libration point in the CR3BP.
    
    Parameters
    ----------
    mu : float
        CR3BP mass parameter (mass ratio of smaller body to total mass)
    L_i : int
        Index of the libration point (1-5)
    
    Returns
    -------
    float
        The reduced mass parameter μ̄
    
    Notes
    -----
    The reduced mass parameter represents the effective gravitational influence
    at the libration point and is used to compute the coefficients of the linearized
    equations of motion. If μ̄ is negative, a warning is issued as this is physically
    unexpected for typical CR3BP configurations.
    """
    L_i = get_lagrange_point(mu, L_i)
    x_L_i = L_i[0]
    mu_bar = mu * np.abs(x_L_i - 1 + mu) ** (-3) + (1 - mu) * np.abs(x_L_i + mu) ** (-3)
    if mu_bar < 0:
        warnings.warn("mu_bar is negative")
    return mu_bar

def _alpha_1(mu, L_i):
    """
    Compute the first eigenvalue coefficient of the libration frame.
    
    This function calculates the α₁ parameter used in deriving the eigenvalues
    of the linearized dynamics around a libration point.
    
    Parameters
    ----------
    mu : float
        CR3BP mass parameter (mass ratio of smaller body to total mass)
    L_i : int
        Index of the libration point (1-5)
    
    Returns
    -------
    float or complex
        The α₁ parameter, which may be complex in certain cases
    
    Notes
    -----
    α₁ is typically positive for collinear libration points, leading to a real
    positive eigenvalue that indicates instability along one direction. If α₁
    is complex, a warning is issued as this might indicate a special case or
    a computational issue.
    """
    mu_bar = _mu_bar(mu, L_i)
    alpha = (mu_bar - 2 + np.emath.sqrt(9*mu_bar**2 - 8*mu_bar)) / 2
    if isinstance(alpha, np.complex128):
        warnings.warn("Alpha 1 is complex")
    return alpha

def _alpha_2(mu, L_i):
    """
    Compute the second eigenvalue coefficient of the libration frame.
    
    This function calculates the α₂ parameter used in deriving the eigenvalues
    of the linearized dynamics around a libration point.
    
    Parameters
    ----------
    mu : float
        CR3BP mass parameter (mass ratio of smaller body to total mass)
    L_i : int
        Index of the libration point (1-5)
    
    Returns
    -------
    float or complex
        The α₂ parameter, which may be complex in certain cases
    
    Notes
    -----
    α₂ is typically negative for collinear libration points, leading to imaginary
    eigenvalues that indicate oscillatory behavior along certain directions.
    If α₂ is complex, a warning is issued as this might indicate a special case or
    a computational issue.
    """
    mu_bar = _mu_bar(mu, L_i)
    alpha = (mu_bar - 2 - np.emath.sqrt(9*mu_bar**2 - 8*mu_bar)) / 2
    if isinstance(alpha, np.complex128):
        warnings.warn("Alpha 2 is complex")
    return alpha

def _beta_1(mu, L_i):
    """
    Compute the first beta coefficient of the libration frame.
    
    This function calculates β₁ = √α₁, which is directly related to the
    real eigenvalues of the system.
    
    Parameters
    ----------
    mu : float
        CR3BP mass parameter (mass ratio of smaller body to total mass)
    L_i : int
        Index of the libration point (1-5)
    
    Returns
    -------
    float or complex
        The β₁ parameter, which may be complex in certain cases
    
    Notes
    -----
    β₁ corresponds to the magnitude of the real eigenvalue that characterizes
    the unstable direction in the dynamics around collinear libration points.
    """
    beta = np.emath.sqrt(_alpha_1(mu, L_i))
    if isinstance(beta, np.complex128):
        warnings.warn("Beta 1 is complex")
    return beta

def _beta_2(mu, L_i):
    """
    Compute the second beta coefficient of the libration frame.
    
    This function calculates β₂ = √α₂, which is directly related to the
    imaginary eigenvalues of the system.
    
    Parameters
    ----------
    mu : float
        CR3BP mass parameter (mass ratio of smaller body to total mass)
    L_i : int
        Index of the libration point (1-5)
    
    Returns
    -------
    float or complex
        The β₂ parameter, which is typically imaginary
    
    Notes
    -----
    β₂ is typically imaginary for collinear libration points and corresponds
    to the frequency of oscillation in the center directions.
    """
    beta = np.emath.sqrt(_alpha_2(mu, L_i))
    if isinstance(beta, np.complex128):
        warnings.warn("Beta 2 is complex")
    return beta

def _tau(mu, L_i):
    """
    Compute the tau coefficient of the libration frame.
    
    This function calculates the τ coefficient used in constructing eigenvectors
    for the linearized system around a libration point.
    
    Parameters
    ----------
    mu : float
        CR3BP mass parameter (mass ratio of smaller body to total mass)
    L_i : int
        Index of the libration point (1-5)
    
    Returns
    -------
    float or complex
        The τ coefficient for the eigenvectors
    
    Notes
    -----
    The τ parameter is used to normalize and construct eigenvectors in a
    consistent way, ensuring they correctly represent the dynamics of the system.
    """
    lambda_1, lambda_2, nu_1, nu_2 = _libration_frame_eigenvalues(mu, L_i)
    return - (nu_1 **2 + _a(mu, L_i)) / (2*nu_1)

def _sigma(mu, L_i):
    """
    Compute the sigma coefficient of the libration frame.
    
    This function calculates the σ coefficient used in constructing eigenvectors
    for the linearized system around a libration point.
    
    Parameters
    ----------
    mu : float
        CR3BP mass parameter (mass ratio of smaller body to total mass)
    L_i : int
        Index of the libration point (1-5)
    
    Returns
    -------
    float or complex
        The σ coefficient for the eigenvectors
    
    Notes
    -----
    The σ parameter is used to normalize and construct eigenvectors in a
    consistent way, ensuring they correctly represent the dynamics of the system.
    """
    lambda_1, lambda_2, nu_1, nu_2 = _libration_frame_eigenvalues(mu, L_i)
    return 2 * lambda_1 / (lambda_1**2 + _b(mu, L_i))

def _a(mu, L_i):
    """
    Compute the 'a' coefficient of the libration frame.
    
    This function calculates the 'a' coefficient used in the linearized 
    equations of motion around a libration point.
    
    Parameters
    ----------
    mu : float
        CR3BP mass parameter (mass ratio of smaller body to total mass)
    L_i : int
        Index of the libration point (1-5)
    
    Returns
    -------
    float
        The 'a' coefficient (2μ̄ + 1)
    
    Notes
    -----
    The 'a' coefficient represents terms in the linearized equations that
    contribute to the overall structure of the eigenvalue problem.
    """
    return 2 * _mu_bar(mu, L_i) + 1

def _b(mu, L_i):
    """
    Compute the 'b' coefficient of the libration frame.
    
    This function calculates the 'b' coefficient used in the linearized 
    equations of motion around a libration point.
    
    Parameters
    ----------
    mu : float
        CR3BP mass parameter (mass ratio of smaller body to total mass)
    L_i : int
        Index of the libration point (1-5)
    
    Returns
    -------
    float
        The 'b' coefficient (μ̄ - 1)
    
    Notes
    -----
    The 'b' coefficient represents terms in the linearized equations that
    contribute to the overall structure of the eigenvalue problem.
    """
    return _mu_bar(mu, L_i) - 1

def _eig_decomp(A, discrete):
    """
    Perform eigendecomposition of a matrix and classify eigenvalues/eigenvectors.
    
    This function computes the eigendecomposition of the given matrix and classifies
    the eigenvalues and eigenvectors into stable, unstable, and center subspaces
    based on eigenvalue characteristics.
    
    Parameters
    ----------
    A : array_like
        Square matrix to decompose, typically a state transition matrix or
        Jacobian of the system
    discrete : int
        Flag indicating the classification criterion:
        * 1: discrete-time system (eigenvalues classified by magnitude relative to 1)
        * 0: continuous-time system (eigenvalues classified by sign of real part)
    
    Returns
    -------
    sn : ndarray
        Array of eigenvalues in the stable subspace
    un : ndarray
        Array of eigenvalues in the unstable subspace
    cn : ndarray
        Array of eigenvalues in the center subspace
    Ws : ndarray
        Matrix whose columns are eigenvectors spanning the stable subspace
    Wu : ndarray
        Matrix whose columns are eigenvectors spanning the unstable subspace
    Wc : ndarray
        Matrix whose columns are eigenvectors spanning the center subspace
    
    Notes
    -----
    For discrete-time systems (discrete=1), the classification is:
    - Stable: |λ| < 1-δ
    - Unstable: |λ| > 1+δ
    - Center: 1-δ ≤ |λ| ≤ 1+δ
    
    For continuous-time systems (discrete=0), the classification is:
    - Stable: Re(λ) < 0
    - Unstable: Re(λ) > 0
    - Center: Re(λ) = 0
    
    where δ is a small tolerance (1e-4 by default).
    
    The function normalizes eigenvectors by dividing by the first non-zero element
    and removes small numerical artifacts in the computed values.
    """
    A = np.asarray(A, dtype=np.complex128)
    M = A.shape[0]
    delta = 1e-4
    
    # 1) Eigenvalues/eigenvectors
    eig_vals, V = np.linalg.eig(A)
    
    # 2) Zero out small imaginary parts in eigenvalues
    eig_vals = np.array([_zero_small_imag_part(ev) for ev in eig_vals])
    
    # Prepare storage
    sn = []
    un = []
    cn = []
    Ws_list = []
    Wu_list = []
    Wc_list = []
    
    # 3) Loop over columns, replicate pivot logic
    for k in range(M):
        val = eig_vals[k]
        vec = V[:, k]
        
        # a) find pivot (first nonzero element)
        jj = 0
        while jj < M and abs(vec[jj]) < 1e-14:
            jj += 1
        if jj < M:
            pivot = vec[jj]
            if abs(pivot) > 1e-14:
                vec = vec / pivot
        
        # b) remove small real/imag parts
        vec = _remove_infinitesimals_array(vec, tol=1e-14)
        
        # c) classification
        if discrete == 1:
            # check magnitude vs 1 ± delta
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
            # continuous case (not shown, but same idea)
            if val.real < 0:
                sn.append(val)
                Ws_list.append(vec)
            elif val.real > 0:
                un.append(val)
                Wu_list.append(vec)
            else:
                cn.append(val)
                Wc_list.append(vec)
    
    # 4) Convert to arrays
    sn = np.array(sn, dtype=np.complex128)
    un = np.array(un, dtype=np.complex128)
    cn = np.array(cn, dtype=np.complex128)
    
    Ws = np.column_stack(Ws_list) if Ws_list else np.zeros((M,0), dtype=np.complex128)
    Wu = np.column_stack(Wu_list) if Wu_list else np.zeros((M,0), dtype=np.complex128)
    Wc = np.column_stack(Wc_list) if Wc_list else np.zeros((M,0), dtype=np.complex128)
    
    return sn, un, cn, Ws, Wu, Wc

def _surface_of_section(X, T, mu, M=1, C=1):
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
    RES = 50

    # Determine the shift d based on M
    if M == 1:
        d = -mu
    elif M == 2:
        d = 1 - mu
    elif M == 0:
        d = 0
    else:
        raise ValueError("Invalid value of M. Must be 0, 1, or 2.")
    
    X = np.array(X, copy=True)  # Copy to avoid modifying the original data
    T = np.array(T)
    n_rows, n_cols = X.shape
    
    # Shift the x-coordinate by subtracting d
    X[:, 0] = X[:, 0] - d

    # Prepare lists to hold crossing states and times
    Xy0_list = []
    Ty0_list = []
    
    # DEBUG: Print trajectory information
    # print(f"DEBUG: Trajectory shape: {X.shape}, min x: {X[:, 0].min()}, max x: {X[:, 0].max()}")
    # print(f"DEBUG: Min y: {X[:, 1].min()}, max y: {X[:, 1].max()}")
    
    # Count potential crossings for debugging
    sign_changes = 0
    y_condition_failures = 0

    if M == 1:
        # For M == 1, use the original data points.
        for k in range(n_rows - 1):
            # Check if there is a sign change in the x-coordinate
            if abs(np.sign(X[k, 0]) - np.sign(X[k+1, 0])) > 0:
                sign_changes += 1
                # Check the condition on y (C*y >= 0)
                if np.sign(C * X[k, 1]) >= 0:
                    # Choose the point with x closer to zero
                    K = k if abs(X[k, 0]) < abs(X[k+1, 0]) else k+1
                    Xy0_list.append(X[K, :])
                    Ty0_list.append(T[k])
                else:
                    y_condition_failures += 1
    elif M == 2:
        # For M == 2, refine the crossing using interpolation.
        for k in range(n_rows - 1):
            if abs(np.sign(X[k, 0]) - np.sign(X[k+1, 0])) > 0:
                sign_changes += 1
                # Interpolate between the two points with increased resolution.
                dt_segment = abs(T[k] - T[k+1]) / RES
                XX, TT = _interpolate(X[k:k+2, :], T[k:k+2], dt_segment)
                # Look through the interpolated points for the crossing
                found_valid_crossing = False
                for kk in range(len(TT) - 1):
                    if abs(np.sign(XX[kk, 0]) - np.sign(XX[kk+1, 0])) > 0:
                        if np.sign(C * XX[kk, 1]) >= 0:
                            K = kk if abs(XX[kk, 0]) < abs(XX[kk+1, 0]) else kk+1
                            Xy0_list.append(XX[K, :])
                            Ty0_list.append(TT[K])
                            found_valid_crossing = True
                        else:
                            y_condition_failures += 1
                
                if not found_valid_crossing:
                    # print(f"DEBUG: No valid crossing found after interpolation at index {k}")
                    # print(f"DEBUG: X[k] = {X[k]}, X[k+1] = {X[k+1]}")
                    pass
    else:
        raise ValueError("Unsupported value for M")
    
    # DEBUG: Print crossing statistics
    # print(f"DEBUG: Found {sign_changes} sign changes, {y_condition_failures} y-condition failures")
    # print(f"DEBUG: Final crossing count: {len(Xy0_list)}")
    
    # Convert lists to arrays
    Xy0 = np.array(Xy0_list)
    Ty0 = np.array(Ty0_list)
    
    return Xy0, Ty0

