import numba
import numpy as np
import sympy as sp


@numba.njit(fastmath=True, cache=True)
def jacobi_integral(state, mu):
    x, y, z, vx, vy, vz = state

    mu_1 = 1 - mu
    mu_2 = mu

    r1 = np.sqrt((x + mu_2)**2 + y**2 + z**2)
    r2 = np.sqrt((x - mu_1)**2 + y**2 + z**2)

    U = - mu_1 / r1 - mu_2 / r2 - 1 / 2 * mu_1 * mu_2

    U_eff = - 1 / 2 * (x**2 + y**2) + U

    return - (vx**2 + vy**2 + vz**2) + 2 * U_eff

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
