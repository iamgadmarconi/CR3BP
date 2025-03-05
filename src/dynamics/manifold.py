import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm

from .corrector import compute_stm
from .propagator import propagate_crtbp

def compute_manifold(x0, T, mu, stbl=1, direction=1, forward=1, step=0.02):
    """
    Computes and plots the stable manifold of a dynamical system.
    
    Parameters:
        x0 : initial condition (could be an array or any structure required by orbitman)
        T  : parameter used in orbitman
        
    Returns:
        ysos  : list of the second component from the Poincaré section (index 1 in Python)
        ydsos : list of the fifth component from the Poincaré section (index 4 in Python)
    """
    ysos = []
    ydsos = []
    xW_list = []
    tW_list = []
    
    # Loop over fractional values from 0 to 0.98 in steps of 0.02
    for frac in tqdm(np.arange(0, 1.0, step), desc="Computing manifold"):  
    # for frac_idx, frac in enumerate(np.arange(0, 1.0, step)):
        # print(f"\nDEBUG: Processing fraction {frac} (index {frac_idx})")
        
        # Get the initial condition on the manifold
        x0W = compute_manifold_section(x0, T, frac, stbl, direction, mu, forward=forward)
        # print(f"DEBUG: Initial condition x0W shape: {x0W.shape}, values: {x0W.flatten()}")
        
        # Define integration time
        tf = 0.7 * (2 * np.pi)
        # Integrate the trajectory starting from x0W over time tf.
        # Ensure x0W is flattened to 1D array
        x0W_flat = x0W.flatten().astype(np.float64)
        # print(f"x0W_flat: {x0W_flat}")
        # Call propagate_crtbp with correct parameter order and get the solution object
        sol = propagate_crtbp(x0W_flat, 0.0,tf, mu, forward=forward)
        
        # Extract state and time vectors from solution object
        # state values are in sol.y with shape (state_dim, n_points), so transpose to (n_points, state_dim)
        xW = sol.y.T
        tW = sol.t
        # print(f"DEBUG: Propagated trajectory shape: {xW.shape}")
        
        xW_list.append(xW)
        tW_list.append(tW)

        # Compute the Poincaré section (sos) at a specified surface (here using 2)
        # print(f"DEBUG: Calling surface_of_section with M=2")
        Xy0, Ty0 = surface_of_section(xW, tW, mu, 2)
        if len(Xy0) == 0:
            # print(f"WARNING: Xy0 is empty for fraction {frac}")
            # Skip this iteration to avoid IndexError
            continue
            
        Xy0 = Xy0.flatten()
        # print(Xy0)
        ysos.append(Xy0[1])
        ydsos.append(Xy0[4])
    
    return ysos, ydsos, xW_list, tW_list

def compute_manifold_section(x0, T, frac, stbl, direction, mu, NN=1, forward=1):
    """
    Python equivalent of the MATLAB orbitman function, fixed to match
    the compute_stm indexing convention.

    Parameters
    ----------
    x0 : array_like
        Initial reference point on the 3D periodic orbit (6D state).
    T : float
        Period of the 3D orbit in nondimensional CR3BP time.
    frac : float
        Fraction (0 to 1) along the orbit at which to compute the manifold.
    stbl : {+1, -1}
        +1 for stable manifold, -1 for unstable manifold.
    direction : {+1, -1}
        +1 for the "positive" branch, -1 for the "negative" branch.
    mu : float
        Mass ratio in the CR3BP.
    NN : int, optional
        If stable/unstable subspace is >1D, pick the NN-th real eigendirection.
        Default is 1.
    forward : {+1, -1}, optional
        Direction of time integration for compute_stm; default +1.

    Returns
    -------
    x0W : ndarray, shape (6,)
        A 6D state on the chosen (un)stable manifold of the 3D periodic orbit.
    """
    # 1) Integrate to get monodromy and the full STM states
    xx, tt, phi_T, PHI = compute_stm(x0, mu, T, forward=forward)
    print(f"DEBUG: phi_T: {phi_T}")
    # 2) Decompose the final monodromy to get stable/unstable eigenvectors
    sn, un, cn, y1Ws, y1Wu, y1Wc = eig_decomp(phi_T, discrete=1)

    # 3) Collect real eigen-directions for stable set
    snreal_vals = []
    snreal_vecs = []
    for k in range(len(sn)):
        if np.isreal(sn[k]):
            snreal_vals.append(sn[k])
            snreal_vecs.append(y1Ws[:, k])

    # 4) Collect real eigen-directions for unstable set
    unreal_vals = []
    unreal_vecs = []
    for k in range(len(un)):
        if np.isreal(un[k]):
            unreal_vals.append(un[k])
            unreal_vecs.append(y1Wu[:, k])

    # Convert lists to arrays
    snreal_vals = np.array(snreal_vals, dtype=np.complex128)
    unreal_vals = np.array(unreal_vals, dtype=np.complex128)
    snreal_vecs = (np.column_stack(snreal_vecs) 
                   if len(snreal_vecs) else np.zeros((6, 0), dtype=np.complex128))
    unreal_vecs = (np.column_stack(unreal_vecs) 
                   if len(unreal_vecs) else np.zeros((6, 0), dtype=np.complex128))

    # 5) Select the NN-th real eigendirection (MATLAB is 1-based, Python is 0-based)
    col_idx = NN - 1
    WS = snreal_vecs[:, col_idx] if snreal_vecs.shape[1] > col_idx >= 0 else None
    WU = unreal_vecs[:, col_idx] if unreal_vecs.shape[1] > col_idx >= 0 else None

    if (WS is None or WS.size == 0) and (WU is None or WU.size == 0):
        raise ValueError("No real eigen-directions found or invalid NN index.")

    # 6) Find the row index for t ~ frac*T
    #    We'll pick the closest time in 'tt' to frac*T:
    mfrac = _totime(tt, frac * T)  # integer index

    # 7) Reshape PHI to get the 6x6 STM at that time.
    #    By the convention in compute_stm, PHI[i, :36] is the flattened 6x6.
    phi_frac_flat = PHI[mfrac, :36]  # first 36 columns
    phi_frac = phi_frac_flat.reshape((6, 6))

    # 8) Decide stable vs. unstable direction
    if stbl == +1:   # stable manifold
        MAN = direction * (phi_frac @ WS)
    else:            # unstable manifold
        MAN = direction * (phi_frac @ WU)

    # 9) Scale the displacement (hard-coded 1e-6 factor)
    disp_magnitude = np.linalg.norm(MAN[0:3])
    if disp_magnitude < 1e-14:
        disp_magnitude = 1.0
    d = 1e-6 / disp_magnitude

    # 10) Reference orbit state at t=tt[mfrac]
    fracH = xx[mfrac, :].copy()  # shape (6,)

    # 11) Construct the final manifold state
    x0W = fracH + d * MAN.real  # ensure real if there's a tiny imaginary part
    x0W = x0W.flatten()
    # 12) Zero out tiny numerical noise in z, z-dot if near zero
    #     (Matching MATLAB orbitman logic, but using abs() for both signs)
    if abs(x0W[2]) < 1.0e-15:
        x0W[2] = 0.0
    if abs(x0W[5]) < 1.0e-15:
        x0W[5] = 0.0

    return x0W

def eig_decomp(A, discrete):
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

def surface_of_section(X, T, mu, M=1, C=1):
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

def _remove_infinitesimals_in_place(vec, tol=1e-14):
    for i in range(len(vec)):
        re = vec[i].real
        im = vec[i].imag
        if abs(re) < tol:
            re = 0.0
        if abs(im) < tol:
            im = 0.0
        vec[i] = re + 1j*im

def _remove_infinitesimals_array(vec, tol=1e-12):
    vcopy = vec.copy()
    _remove_infinitesimals_in_place(vcopy, tol)
    return vcopy

def _zero_small_imag_part(eig_val, tol=1e-12):
    if abs(eig_val.imag) < tol:
        return complex(eig_val.real, 0.0)
    return eig_val

def _totime(t, tf):
    # Convert t to its absolute values.
    t = np.abs(t)
    # Ensure tf is an array (handles scalar input as well).
    tf = np.atleast_1d(tf)
    
    # Preallocate an array to hold the indices.
    I = np.empty(tf.shape, dtype=int)
    
    # For each target value in tf, find the index in t closest to it.
    for k, target in enumerate(tf):
        diff = np.abs(target - t)
        I[k] = np.argmin(diff)
    
    return I

def _interpolate(x, t, dt=None):
    """
    Re-samples a trajectory x with time stamps t using cubic spline interpolation.
    
    Parameters:
      x  : ndarray of shape (m, n) -- data to interpolate (each column is a variable)
      t  : 1D array of length m   -- original time points
      dt : Either the time step between interpolated points or,
           if dt > 10, the number of points desired.
           If not provided, defaults to 0.05 * 2*pi.
    
    Returns:
      X  : ndarray -- interpolated data with evenly spaced time steps
      T  : 1D array -- new time vector corresponding to X
    """
    t = np.asarray(t)
    x = np.asarray(x)
    
    # Default dt if not provided
    if dt is None:
        dt = 0.05 * 2 * np.pi

    # If dt > 10, then treat dt as number of points (N) and recalc dt
    if dt > 10:
        N = int(dt)
        dt = (np.max(t) - np.min(t)) / (N - 1)
    
    # Adjust time vector if it spans negative and positive values
    NEG = 1 if (np.min(t) < 0 and np.max(t) > 0) else 0
    tt = np.abs(t - NEG * np.min(t))
    
    # Create new evenly spaced time vector for the interpolation domain
    TT = np.arange(tt[0], tt[-1] + dt/10, dt)
    # Recover the correct "arrow of time"
    T = np.sign(t[-1]) * TT + NEG * np.min(t)
    
    # Interpolate each column using cubic spline interpolation
    if x.ndim == 1:
        # For a single-dimensional x, treat as a single column
        cs = CubicSpline(tt, x)
        X = cs(TT)
    else:
        m, n = x.shape
        X = np.zeros((len(TT), n))
        for i in range(n):
            cs = CubicSpline(tt, x[:, i])
            X[:, i] = cs(TT)
    
    return X, T