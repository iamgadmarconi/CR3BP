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
            tW, xW = globalize_manifold(x0W, mu, tf, forward=False)
        else:
            tW, xW = globalize_manifold(x0W, mu, tf, forward=True)

        xW_list.append(xW)
        tW_list.append(tW)

    return xW_list, tW_list

def globalize_manifold(x0W, mu, tf=2*np.pi, forward=True):
    """
    Integrate the manifold forward or backward for time tf.
    """
    sign = +1 if forward else -1
    def ode_func(t, X):
        return sign * np.array(crtbp_accel(X, mu))

    sol = solve_ivp(ode_func, [0, tf], x0W, t_eval=np.linspace(0, tf, 400))
    return sol.t, sol.y.T

def get_manifold_directions(Phi_T):
    """
    Decompose the monodromy matrix Phi_T = Phi(0,T).
    Return the stable, unstable, center eigenvalues/vectors.
    """
    w, v = np.linalg.eig(Phi_T)
    
    # We'll store real eigenvalues < 1 in stable, > 1 in unstable, ~1 in center
    # For the continuous-time system, typically we look at exp(lambda*T).
    # But let's keep it simple: if |lambda|<1 => stable, etc.
    stable_dirs   = []
    stable_eigs   = []
    unstable_dirs = []
    unstable_eigs = []
    center_dirs   = []
    center_eigs   = []
    
    eps = 1e-14
    for i in range(len(w)):
        lam = w[i]
        vec = v[:,i]
        if abs(abs(lam) - 1.0) < eps:
            center_dirs.append(vec)
            center_eigs.append(lam)
        elif abs(lam) < 1.0:
            stable_dirs.append(vec)
            stable_eigs.append(lam)
        else:
            unstable_dirs.append(vec)
            unstable_eigs.append(lam)
    # convert to arrays if needed
    return (np.array(stable_eigs),   np.array(stable_dirs).T,
            np.array(unstable_eigs), np.array(unstable_dirs).T,
            np.array(center_eigs),   np.array(center_dirs).T)

def orbit_manifold(x_traj, Phi_list, t_array, T, frac, stable_vec, dir_sign=+1, scale=1e-6):
    """
    Python equivalent of 'orbitman.m' for a 3D periodic orbit.

    - x_traj: shape (n_steps+1, 6) states from t=0..T
    - Phi_list: list of 6x6 STM from t=0..T
    - t_array: times array shape (n_steps+1,)
    - T: the full period
    - frac: fraction in [0..1], i.e. time = frac * T
    - stable_vec: a single 6D eigenvector (or whichever direction you want)
    - dir_sign: +1 or -1 for branch
    - scale: how big to displace along that direction

    Returns x0W: the manifold initial condition in 6D
    """
    # 1) find the index i where t_array[i] ~ frac*T
    desired_t = frac*T
    i_closest = np.argmin(np.abs(t_array - desired_t))
    # partial STM = Phi(0, t_i)
    Phi_frac = Phi_list[i_closest]
    x_frac = x_traj[i_closest]

    # 2) The manifold direction = dir_sign * Phi(0,t_frac) * stable_vec
    # scale it
    disp_vec = scale * (Phi_frac @ stable_vec)
    
    # 3) The manifold point
    x0W = x_frac + disp_vec

    return np.array(x0W, dtype=np.float64)
