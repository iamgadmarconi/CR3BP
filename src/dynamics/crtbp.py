import numba
import math
import mpmath as mp
import numpy as np

from utils.frames import _alpha_1, _alpha_2, _libration_frame_eigenvalues


mp.mp.dps = 50 

def crtbp_energy(state, mu):
    """
    state: shape (6,) -> [x, y, z, vx, vy, vz]
    Returns scalar energy (Jacobi-like).
    """
    x, y, z, vx, vy, vz = state
    mu1 = 1.0 - mu
    mu2 = mu
    
    r1 = np.sqrt((x + mu2)**2 + y**2 + z**2)
    r2 = np.sqrt((x - mu1)**2 + y**2 + z**2)
    
    kin = 0.5 * (vx*vx + vy*vy + vz*vz)
    pot = -(mu1 / r1) - (mu2 / r2) - 0.5*(x*x + y*y + z*z) - 0.5*mu1*mu2
    return kin + pot

def compute_energy_bounds(mu, case):
    """
    Compute the energy bounds corresponding to a given case (1-5) in the CR3BP, 
    for a specified mass ratio mu in [0, 0.5].
    
    Returns a tuple (E_lower, E_upper) giving the energy range for that case.

    Case 1: E < E1 (below L1 energy - motion bound near one primary only)
    Case 2: E1 <= E < E2 (between L1 and L2 energy - L1 neck open, others closed)
    Case 3: E2 <= E < E3 (between L2 and L3 energy - L1 and L2 open, L3 closed)
    Case 4: E3 <= E < E4 (between L3 and L4/L5 energy - all collinear passages open)
    Case 5: E >= E4 (at or above L4/L5 energy - motion is virtually unrestricted)
    """
    if mu < 0 or mu > 0.5:
        raise ValueError("Mass ratio mu must be between 0 and 0.5 (inclusive).")

    if abs(mu) < 1e-9:  # treat mu == 0 as two-body problem (secondary has zero mass)
        E1 = E2 = E3 = E4 = E5 = -1.5
    else:

        x_L1 = _l1(mu)
        x_L2 = _l2(mu)
        x_L3 = _l3(mu)

        def Omega(x, y, mu):
            r1 = np.sqrt((x + mu)**2 + y**2)
            r2 = np.sqrt((x - 1 + mu)**2 + y**2)
            return 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2
        
        E1 = -Omega(x_L1[0], 0.0, mu)   # energy level at L1
        E2 = -Omega(x_L2[0], 0.0, mu)   # energy level at L2
        E3 = -Omega(x_L3[0], 0.0, mu)   # energy level at L3
        
        E4 = E5 = -1.5
    
    if case == 1:
        return (-math.inf, E1)
    elif case == 2:
        return (E1, E2)
    elif case == 3:
        return (E2, E3)
    elif case == 4:
        return (E3, E4)
    elif case == 5:
        return (E4, math.inf)
    else:
        raise ValueError("Case number must be between 1 and 5.")

def _energy_to_jacobi_constant(E):
    return -2 * E

def hill_region(mu, C, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), n_grid=400):
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(x, y)

    r1 = np.sqrt((X + mu)**2 + Y**2)
    r2 = np.sqrt((X - 1 + mu)**2 + Y**2)

    Omega = (1 - mu) / r1 + mu / r2 + 0.5 * (X**2 + Y**2)

    Z = Omega - C/2

    return X, Y, Z

def _kinetic_energy(state):
    x, y, z, vx, vy, vz = state
    return 1 / 2 * (vx**2 + vy**2 + vz**2)

def energy_integral(state, mu):
    x, y, z, vx, vy, vz = state
    U_eff = _effective_potential(state, mu)
    return 1 / 2 * (vx**2 + vy**2 + vz**2) + U_eff

def jacobi_constant(state, mu):
    x, y, z, vx, vy, vz = state
    U_eff = _effective_potential(state, mu)
    return - (vx**2 + vy**2 + vz**2) - 2 * U_eff

def _effective_potential(state, mu):
    x, y, z, vx, vy, vz = state
    mu_1 = 1 - mu
    mu_2 = mu
    r1 = _primary_distance(state, mu)
    r2 = _secondary_distance(state, mu)
    U = _potential(state, mu)
    U_eff = - 1 / 2 * (x**2 + y**2 + z**2) + U
    return U_eff

def _potential(state, mu):
    x, y, z, vx, vy, vz = state
    mu_1 = 1 - mu
    mu_2 = mu
    r1 = _primary_distance(state, mu)
    r2 = _secondary_distance(state, mu)
    U = - mu_1 / r1 - mu_2 / r2 - 1 / 2 * mu_1 * mu_2
    return U

def _primary_distance(state, mu):
    x, y, z, vx, vy, vz = state
    mu_2 = mu
    r1 = np.sqrt((x + mu_2)**2 + y**2 + z**2)
    return r1

def _secondary_distance(state, mu):
    x, y, z, vx, vy, vz = state
    mu_1 = 1 - mu
    r2 = np.sqrt((x - mu_1)**2 + y**2 + z**2)
    return r2

def libration_points(mu):
    collinear = _collinear_points(mu)
    equilateral = _equilateral_points(mu)
    return collinear, equilateral

def _equilateral_points(mu):
    return _l4(mu), _l5(mu)

def _collinear_points(mu):
    return _l1(mu), _l2(mu), _l3(mu)

def _l1(mu):
    x = mp.findroot(lambda x: _dOmega_dx(x, mu), [-mu + 0.01, 1 - mu - 0.01])
    x = float(x)
    return np.array([x, 0, 0], dtype=np.float64)

def _l2(mu):
    x = mp.findroot(lambda x: _dOmega_dx(x, mu), [1.0, 2.0])
    x = float(x)
    return np.array([x, 0, 0], dtype=np.float64)

def _l3(mu):
    x = mp.findroot(lambda x: _dOmega_dx(x, mu), [-mu - 0.01, -2.0])
    x = float(x)
    return np.array([x, 0, 0], dtype=np.float64)

def _l4(mu):
    x = 1 / 2 - mu
    y = np.sqrt(3) / 2
    return np.array([x, y, 0], dtype=np.float64)

def _l5(mu):
    x = 1 / 2 - mu
    y = -np.sqrt(3) / 2
    return np.array([x, y, 0], dtype=np.float64)

def _dOmega_dx(x, mu):
    r1 = abs(x + mu)
    r2 = abs(x - (1 - mu))
    return x - (1 - mu) * (x + mu) / (r1**3)  -  mu * (x - (1 - mu)) / (r2**3)