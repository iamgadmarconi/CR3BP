import numba
import numpy as np


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
    pass

def _collinear_points(mu):
    pass

