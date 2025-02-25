import numba
import numpy as np
import sympy as sp
from skimage import measure


import numpy as np
import matplotlib.pyplot as plt

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
    x = sp.symbols('x')
    eq = x - (1 - mu) / abs(x + mu)**3 * (x + mu) - mu / abs(x - 1 + mu)**3 * (x - 1 + mu)
    sol = sp.nsolve(eq, -1)

    return np.array([sol, 0, 0], dtype=np.float64)

def _l2(mu):
    x = sp.symbols('x')
    eq = x - (1 - mu) / abs(x + mu)**3 * (x + mu) - mu / abs(x - 1 + mu)**3 * (x - 1 + mu)
    sol = sp.nsolve(eq, 1)

    return np.array([sol, 0, 0], dtype=np.float64)

def _l3(mu):
    x = sp.symbols('x')
    eq = x - (1 - mu) / abs(x + mu)**3 * (x + mu) - mu / abs(x - 1 + mu)**3 * (x - 1 + mu)
    sol = sp.nsolve(eq, 0.5)

    return np.array([sol, 0, 0], dtype=np.float64)

def _l4(mu):
    x = 1 / 2 - mu
    y = np.sqrt(3) / 2
    return np.array([x, y, 0], dtype=np.float64)

def _l5(mu):
    x = 1 / 2 - mu
    y = -np.sqrt(3) / 2
    return np.array([x, y, 0], dtype=np.float64)
