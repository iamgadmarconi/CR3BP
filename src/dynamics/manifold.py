import numpy as np
from scipy.integrate import solve_ivp

from .propagator import crtbp_accel
from .corrector import compute_stm

def generate_manifold(x0, mu, half_period, stable=True, n_steps=30):
    tf = 2 * half_period
    ts, x_traj, Phi_list = compute_stm(x0, mu, tf, store_states=True, n_steps=n_steps)

    Phi_T = Phi_list[-1]

    seigs, sdirs, ueigs, udirs, ceigs, cdirs = get_manifold_directions(Phi_T, tf)

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

def get_manifold_directions(Phi_T, T, eps=1e-12):
    """
    Decompose monodromy matrix Phi_T = Phi(0,T) for a continuous-time system.
    The eigenvalues of Phi_T are e^(lambda_i * T).
    If real(lambda_i) < 0 => stable, real(lambda_i) > 0 => unstable, else center.
    """
    w, v = np.linalg.eig(Phi_T)

    stable_eigs   = []
    stable_dirs   = []
    unstable_eigs = []
    unstable_dirs = []
    center_eigs   = []
    center_dirs   = []

    for i in range(len(w)):
        lam_exp = w[i]   # e^(lambda_i * T)
        vec     = v[:,i]
        # Solve for lambda_i = (1/T) * log(lam_exp) in complex sense:
        # watch out for branch cuts if lam_exp < 0, etc.
        lam = np.log(lam_exp) / T  
        
        if abs(lam.imag) < eps:
            # If nearly real, force it to be real
            lam = lam.real
        
        if abs(lam.real) < eps:
            # center
            center_eigs.append(lam_exp)
            center_dirs.append(vec)
        elif lam.real < 0:
            stable_eigs.append(lam_exp)
            stable_dirs.append(vec)
        else:
            unstable_eigs.append(lam_exp)
            unstable_dirs.append(vec)

    return (np.array(stable_eigs),
            np.array(stable_dirs).T,  # shape (6, #stable)
            np.array(unstable_eigs),
            np.array(unstable_dirs).T, 
            np.array(center_eigs),
            np.array(center_dirs).T
    )

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
