import numpy as np

from utils.frames import (_libration_frame_eigenvectors, 
                        _alpha_1, _alpha_2, _beta_1, _beta_2)             

def general_linear_ic(mu, L_i):
    """
    Returns the rotating-frame initial condition for the linearized CR3BP
    near the libration point L_i, for the chosen coefficients alpha1, alpha2, beta1, beta2.
    """
    # 1) Get the four eigenvectors (u1,u2,w1,w2)
    u1, u2, w1, w2 = _libration_frame_eigenvectors(mu, L_i)
    alpha1 = _alpha_1(mu, L_i)
    alpha2 = _alpha_2(mu, L_i)
    beta1 = _beta_1(mu, L_i)
    beta2 = _beta_2(mu, L_i)

    beta = beta1 + beta2*1j

    term = 2 * np.real(beta * w1)

    z = alpha1 * u1 + alpha2 * u2 + term

    return z
