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