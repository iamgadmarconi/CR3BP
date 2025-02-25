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
    return _collinear_points(mu), _equilateral_points(mu)


def _equilateral_points(mu):
    return _l4(mu), _l5(mu)

def _collinear_points(mu):
    return _l1(mu), _l2(mu), _l3(mu)


def _l1(mu):
    x = sp.symbols('x')
    eq = x**5 + (3-mu)*x**4 + (3-2*mu)*x**3 - mu*x**2 - 2* mu * x - mu
    sol = sp.nsolve(eq, mu/3)

    return sol

def _l2(mu):
    x = sp.symbols('x')
    eq = x - 1 + mu + (1 - mu) / (x - 1)**2 - mu / (x**2)
    sol = sp.nsolve(eq, (mu/3)**(1/3))

    return sol

def _l3(mu):
    x = sp.symbols('x')
    eq = x**5 + (7+mu)*x**4 + (19+6*mu)*x**3 - (24+13*mu)*x**2 - 2*(6+7*mu)*x + 7*mu
    sol = sp.nsolve(eq, -7/12*mu)

    return sol

def _l4(mu):
    return mu

def _l5(mu):
    return 1 - mu


print(_l1(0.0121505856096261))
print(_l2(0.0121505856096261))
print(_l3(0.0121505856096261))

