import numpy as np
import warnings

from dynamics.manifolds.utils import _remove_infinitesimals_array, _zero_small_imag_part, _interpolate


def _libration_frame_eigenvalues(mu, L_i):
    """
    Compute the eigenvalues of the libration frame.

    Args:
        mu: CR3BP mass parameter (dimensionless)
        L_i: Libration point coordinates in dimensionless units

    Returns:
        eigenvalues: Numpy array of eigenvalues
    """
    mu_bar = _mu_bar(mu, L_i)
    alpha_1 = _alpha_1(mu, L_i)
    alpha_2 = _alpha_2(mu, L_i)

    eig1 = np.sqrt(alpha_1)
    eig2 = np.emath.sqrt(-alpha_2)

    return eig1, -eig1, eig2, -eig2

def _libration_frame_eigenvectors(mu, L_i, orbit_type="lyapunov"):
    """
    Compute the eigenvectors of the libration frame.
    """
    mu_bar = _mu_bar(mu, L_i)
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
    Compute the reduced mass parameter.
    """
    x_L_i = L_i[0]
    mu_bar = mu * np.abs(x_L_i - 1 + mu) ** (-3) + (1 - mu) * np.abs(x_L_i + mu) ** (-3)
    if mu_bar < 0:
        warnings.warn("mu_bar is negative")
    return mu_bar

def _alpha_1(mu, L_i):
    """
    Compute the first eigenvalue of the libration frame.
    """
    mu_bar = _mu_bar(mu, L_i)
    alpha = (mu_bar - 2 + np.emath.sqrt(9*mu_bar**2 - 8*mu_bar)) / 2
    if isinstance(alpha, np.complex128):
        warnings.warn("Alpha 1 is complex")
    return alpha

def _alpha_2(mu, L_i):
    """
    Compute the second eigenvalue of the libration frame.
    """
    mu_bar = _mu_bar(mu, L_i)
    alpha = (mu_bar - 2 - np.emath.sqrt(9*mu_bar**2 - 8*mu_bar)) / 2
    if isinstance(alpha, np.complex128):
        warnings.warn("Alpha 2 is complex")
    return alpha

def _beta_1(mu, L_i):
    """
    Compute the first beta coefficient of the libration frame.
    """
    beta = np.emath.sqrt(_alpha_1(mu, L_i))
    if isinstance(beta, np.complex128):
        warnings.warn("Beta 1 is complex")
    return beta

def _beta_2(mu, L_i):
    """
    Compute the second beta coefficient of the libration frame.
    """
    beta = np.emath.sqrt(_alpha_2(mu, L_i))
    if isinstance(beta, np.complex128):
        warnings.warn("Beta 2 is complex")
    return beta

def _tau(mu, L_i):
    """
    Compute the tau coefficient of the libration frame.
    """
    lambda_1, lambda_2, nu_1, nu_2 = _libration_frame_eigenvalues(mu, L_i)
    return - (nu_1 **2 + _a(mu, L_i)) / (2*nu_1)

def _sigma(mu, L_i):
    """
    Compute the sigma coefficient of the libration frame.
    """
    lambda_1, lambda_2, nu_1, nu_2 = _libration_frame_eigenvalues(mu, L_i)
    return 2 * lambda_1 / (lambda_1**2 + _b(mu, L_i))

def _a(mu, L_i):
    """
    Compute the a coefficient of the libration frame.
    """
    return 2 * _mu_bar(mu, L_i) + 1

def _b(mu, L_i):
    """
    Compute the b coefficient of the libration frame.
    """
    return _mu_bar(mu, L_i) - 1

def _eig_decomp(A, discrete):
    """
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
    Compute the surface-of-section for the CR3BP at the x-d=0 crossing
    with the condition C*y >= 0.

    Parameters:
      X : ndarray with shape (n, state_dim)
          The state trajectory (each row is a state vector; first column is x, second is y, etc.)
      T : 1D array of length n
          Time stamps corresponding to the state trajectory.
      M : int, optional
          Determines which body is used for the offset:
            - M = 2 -> d = 1-mu (smaller mass, M2)
            - M = 1 -> d = -mu   (larger mass, M1)
            - M = 0 -> d = 0     (center-of-mass)
          Default is 1.
      C : int, optional
          Crossing condition on y:
            - C = 1: accept crossings with y >= 0
            - C = -1: accept crossings with y <= 0
            - C = 0: accept both sides
          Default is 1.
    
    Returns:
      Xy0 : ndarray
            Array of state vectors at the crossing points.
      Ty0 : ndarray
            Array of times corresponding to the crossings.
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

