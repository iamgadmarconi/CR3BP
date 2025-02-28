import numpy as np
from scipy.integrate import solve_ivp

from .propagator import crtbp_accel
from .corrector import compute_stm

def generate_manifold(x0, mu, half_period, stable=True, n_steps=30):
    tf = 2 * half_period
    ts, x_traj, Phi_list = compute_stm(x0, mu, tf, store_states=True, n_steps=n_steps)

    Phi_T = Phi_list[-1]

    seigs, sdirs, ueigs, udirs, ceigs, cdirs = get_manifold_directions(Phi_T)

    if stable and len(sdirs) > 0:
        direction = sdirs[:,0]  
    elif (not stable) and len(udirs) > 0:
        direction = udirs[:,0]
    else:
        raise ValueError("No stable or unstable directions found")

    frac_values = np.linspace(0, 1, n_steps)

    xW_list = []
    tW_list = []

    for frac in frac_values:
        x0W = orbit_manifold(x_traj, Phi_list, ts, tf, frac, direction)

        if stable:
            tW, xW = integrate_manifold(x0W, mu, 0.7*tf, forward=False)
        else:
            tW, xW = integrate_manifold(x0W, mu, 0.7*tf, forward=True)

        xW_list.append(xW)
        tW_list.append(tW)

    return xW_list, tW_list

def integrate_manifold(x0W, mu, tf, forward=True):
    """
    Integrate the manifold forward or backward for time tf.
    """
    sign = +1 if forward else -1
    def ode_func(t, X):
        return sign * np.array(crtbp_accel(X, mu))

    sol = solve_ivp(ode_func, [0, tf], x0W, t_eval=np.linspace(0, tf, 400))
    return sol.t, sol.y.T

def get_manifold_directions(Phi_T, delta=1e-4):
    """
    Replicate mani(A, discrete=1) from Ross's code.
    That is, we classify each eigenvalue of A = Phi_T as:
      if |lambda| < 1 - delta => stable
      if |lambda| > 1 + delta => unstable
      else => center
    """
    w, V = np.linalg.eig(Phi_T)   # w: eigenvalues, V: columns are eigenvectors
    Ws_list, Wu_list, Wc_list = [], [], []
    sn_list, un_list, cn_list = [], [], []

    for i in range(len(w)):
        lam_exp = w[i]
        # Magnitude of eigenvalue
        mag = abs(lam_exp)
        # Identify the subspace
        if mag < (1.0 - delta):
            # stable
            sn_list.append(lam_exp)
            # Ross divides the eigenvector by whichever nonzero element
            # so that the first non-tiny component is 1. We can do the same:
            vec = V[:,i]
            # find first nonzero entry
            idx_first_nonzero = np.argmax(np.abs(vec) > 1e-14)
            if np.abs(vec[idx_first_nonzero]) > 1e-14:
                vec = vec / vec[idx_first_nonzero]
            vec = remove_infinitesimals(vec)
            Ws_list.append(vec)
        elif mag > (1.0 + delta):
            # unstable
            un_list.append(lam_exp)
            vec = V[:,i]
            idx_first_nonzero = np.argmax(np.abs(vec) > 1e-14)
            if np.abs(vec[idx_first_nonzero]) > 1e-14:
                vec = vec / vec[idx_first_nonzero]
            vec = remove_infinitesimals(vec)
            Wu_list.append(vec)
        else:
            # center
            cn_list.append(lam_exp)
            vec = V[:,i]
            idx_first_nonzero = np.argmax(np.abs(vec) > 1e-14)
            if np.abs(vec[idx_first_nonzero]) > 1e-14:
                vec = vec / vec[idx_first_nonzero]
            vec = remove_infinitesimals(vec)
            Wc_list.append(vec)
    
    # Convert to arrays
    sn = np.array(sn_list, dtype=complex)
    Ws = np.array(Ws_list).T if len(Ws_list)>0 else np.zeros((len(w),0))
    un = np.array(un_list, dtype=complex)
    Wu = np.array(Wu_list).T if len(Wu_list)>0 else np.zeros((len(w),0))
    cn = np.array(cn_list, dtype=complex)
    Wc = np.array(Wc_list).T if len(Wc_list)>0 else np.zeros((len(w),0))

    return sn, un, cn, Ws, Wu, Wc

def remove_infinitesimals(vec, tol=1e-14):
    real_part = np.real(vec)
    imag_part = np.imag(vec)
    # zero out small real or imaginary parts
    real_part[np.abs(real_part) < tol] = 0.0
    imag_part[np.abs(imag_part) < tol] = 0.0
    # reassemble
    return real_part + 1j*imag_part

def orbit_manifold(x_traj, Phi_list, t_array, T, frac, eig_vec, dir_sign=+1):
    """
    Python equivalent of orbitman for 3D orbits. 
    We choose to normalize the first 3 components to 1e-6.
    """
    desired_t = frac * T
    i_closest = np.argmin(np.abs(t_array - desired_t))
    x_frac = x_traj[i_closest]          # the 'reference' point on the orbit
    Phi_frac = Phi_list[i_closest]      # partial STM

    # multiply the eigenvector
    raw_vec = Phi_frac @ eig_vec

    # Possibly scale so that the displacement in position is ~ 1e-6
    # (or 1e-7, or 200 km in dimensioned units—whatever you prefer!)
    pos_norm = np.linalg.norm(raw_vec[0:3])
    if pos_norm < 1e-14:
        pos_norm = 1e-14
    scale = 1e-6 / pos_norm
    disp_vec = dir_sign * scale * raw_vec

    x0W = x_frac + disp_vec
    return np.array(x0W, dtype=np.float64)
