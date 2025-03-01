import numpy as np
from .corrector import compute_stm


def compute_manifold(x0, T, frac, stbl, direction, mu, NN=1):
    """
    Python equivalent of the MATLAB orbitman function.

    Parameters
    ----------
    x0 : array_like
        Initial reference point on the 3D periodic orbit (6D state in CR3BP).
    T : float
        Period of the 3D orbit in nondimensional CR3BP time.
    frac : float
        Fraction (0 to 1) along the orbit (beyond x0) at which to compute the manifold.
    stbl : {+1, -1}
        +1 for stable manifold, -1 for unstable manifold.
    direction : {+1, -1}
        +1 for the "positive" branch of the manifold, -1 for the "negative" branch.
    NN : int, optional
        If the stable/unstable subspace is more than 1-dimensional,
        pick the NN-th eigen-direction. Default is 1 (the first direction).
    
    Returns
    -------
    x0W : ndarray
        A 6D state on the chosen (un)stable manifold of the 3D periodic orbit.
    """
    # 1) Get monodromy and orbit info
    xx, tt, phi_T, PHI = compute_stm(x0, mu, T)
    # 2) Get stable / unstable eigenvalues and eigenvectors
    sn, un, cn, y1Ws, y1Wu, y1Wc = eig_decomp(phi_T, discrete=1)
    # print(f'y1Wc: {y1Wc}')
    # 3) Extract real eigen-directions from stable set
    snreal_vals = []
    snreal_vecs = []
    for k in range(len(sn)):
        if np.isreal(sn[k]):
            snreal_vals.append(sn[k])
            snreal_vecs.append(y1Ws[:, k])
    
    # 4) Extract real eigen-directions from unstable set
    unreal_vals = []
    unreal_vecs = []
    for k in range(len(un)):
        if np.isreal(un[k]):
            unreal_vals.append(un[k])
            unreal_vecs.append(y1Wu[:, k])
    
    # Convert lists to arrays
    snreal_vals = np.array(snreal_vals, dtype=np.complex128)
    unreal_vals = np.array(unreal_vals, dtype=np.complex128)
    snreal_vecs = np.column_stack(snreal_vecs) if snreal_vecs else np.array([])
    unreal_vecs = np.column_stack(unreal_vecs) if unreal_vecs else np.array([])
    
    # 5) Select the NN-th real eigendirection (column)
    #    Note that MATLAB indexing is 1-based, Python is 0-based:
    col_idx = NN - 1  
    WS = snreal_vecs[:, col_idx] if snreal_vecs.size > 0 else None
    WU = unreal_vecs[:, col_idx] if unreal_vecs.size > 0 else None
    
    if WS is None and WU is None:
        raise ValueError("No real eigen-directions found or invalid NN index.")
    
    # 6) Find the time index corresponding to frac*T
    mfrac = _totime(tt, frac * T)
    
    # 7) Reshape PHI to get the 6x6 STM at that time
    #    PHI is assumed to have shape (num_times, 36) with each 6x6 flattened
    phi_slice = PHI[mfrac, 6:]
    phi_frac = phi_slice.reshape(6, 6)
    
    # 8) Depending on stable/unstable selection, compute direction
    if stbl == +1:
        # stable manifold
        MAN = direction * phi_frac @ WS
    else:
        # unstable manifold
        MAN = direction * phi_frac @ WU
    
    # 9) Scale the displacement
    #    Barden's suggestion: ~200 km for Sun-Earth, but here it's coded 1.e-6
    disp_magnitude = np.linalg.norm(MAN[0:3])
    if disp_magnitude < 1e-14:
        # Avoid dividing by zero if the vector is negligible
        disp_magnitude = 1.0
    d = 1.0e-6 / disp_magnitude
    
    # 10) Reference orbit point at t = frac*T
    fracH = xx[mfrac, :].copy()
    
    # 11) Create the final manifold point
    x0W = fracH + d * MAN
    x0W = x0W.reshape(6, 1)
    # print(f'x0W: {x0W}')
    # 12) Remove tiny numerical noise in z, z-dot if near zero
    if abs(x0W[2]) < 1.0e-15:
        x0W[2] = 0.0
    if abs(x0W[5]) < 1.0e-15:
        x0W[5] = 0.0
    
    # In MATLAB, there's a global FORWARD set to -stbl; in Python,
    # you might just return that value or handle it externally.
    FORWARD = -stbl  # or ignore if you don't need this global variable
    
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
        while jj < M and abs(vec[jj]) < 1e-16:
            jj += 1
        if jj < M:
            pivot = vec[jj]
            if abs(pivot) > 1e-16:
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

def _remove_infinitesimals_in_place(vec, tol=1e-14):
    for i in range(len(vec)):
        re = vec[i].real
        im = vec[i].imag
        if abs(re) < tol:
            re = 0.0
        if abs(im) < tol:
            im = 0.0
        vec[i] = re + 1j*im

def _remove_infinitesimals_array(vec, tol=1e-14):
    vcopy = vec.copy()
    _remove_infinitesimals_in_place(vcopy, tol)
    return vcopy

def _zero_small_imag_part(eig_val, tol=1e-14):
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
