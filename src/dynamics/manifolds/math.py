def _eig_decomp(A, discrete):
    """
    Python version matching MATLAB's 'mani(A, discrete)'
    as closely as possible.
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

