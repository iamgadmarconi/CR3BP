"""
State Transition Matrix (STM) computations for the CR3BP.

This module provides functionality for computing and analyzing state transition 
matrices in the Circular Restricted Three-Body Problem (CR3BP). The state 
transition matrix maps how small perturbations in initial conditions evolve 
over time, which is crucial for:

1. Stability analysis of periodic orbits
2. Differential correction and trajectory targeting
3. Computing Lyapunov exponents and chaos indicators
4. Constructing invariant manifolds

The implementation in this module focuses on robust numerical integration of
the variational equations alongside the CR3BP equations of motion.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .equations import variational_equations


def compute_stm(x0, mu, tf, forward=1, **solve_kwargs):
    """
    Compute the State Transition Matrix (STM) for the CR3BP.
    
    This function integrates the combined CR3BP equations of motion and 
    variational equations from t=0 to t=tf to obtain the state transition 
    matrix Φ(tf, 0). The implementation mirrors MATLAB's var3D approach
    for compatibility with existing methods.
    
    Parameters
    ----------
    x0 : array_like
        Initial state vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    tf : float
        Final integration time (must be positive)
    forward : int, optional
        Direction of integration (1 for forward, -1 for backward). Default is 1.
    **solve_kwargs
        Additional keyword arguments passed to scipy.integrate.solve_ivp,
        such as 'rtol', 'atol', or 'method'
    
    Returns
    -------
    x : ndarray
        Array of shape (n_times, 6) containing the state trajectory
    t : ndarray
        Array of shape (n_times,) containing the time points
        (If forward=-1, times are negated to reflect backward integration)
    phi_T : ndarray
        The 6x6 state transition matrix Φ(tf, 0) at the final time
    PHI : ndarray
        Array of shape (n_times, 42) containing the full integrated
        solution at each time point, where each row is [flattened STM(36), state(6)]
    
    Notes
    -----
    The state transition matrix Φ(t, t0) maps perturbations in the initial state
    to perturbations at time t according to: δx(t) = Φ(t, t0)·δx(t0).
    
    The STM is initialized as the 6x6 identity matrix at t=0, representing
    the fact that initially, a perturbation in any direction affects only that
    direction with unit magnitude.
    
    For numerical reasons, the function always integrates forward in time
    (from 0 to |tf|), but controls the direction of the dynamics through the
    'forward' parameter, which is multiplied by the derivatives.
    """

    # Build initial 42-vector in MATLAB ordering: [flattened STM, state]
    PHI0 = np.zeros(42, dtype=np.float64)
    # The first 36 = identity matrix
    PHI0[:36] = np.eye(6, dtype=np.float64).ravel()
    # The last 6 = x0
    PHI0[36:] = x0

    # Set default solver tolerances if not provided
    if 'rtol' not in solve_kwargs:
        solve_kwargs['rtol'] = 3e-14
    if 'atol' not in solve_kwargs:
        solve_kwargs['atol'] = 1e-14

    def ode_fun(t, y):
        # Calls our Numba-accelerated function
        return variational_equations(t, y, mu, forward)

    # Integrate from 0 to tf
    t_span = (0.0, tf)
    sol = solve_ivp(ode_fun, t_span, PHI0, **solve_kwargs)

    # Possibly flip time if forward==-1, to mirror MATLAB's t=FORWARD*t
    if forward == -1:
        sol.t = -sol.t  # so we see times from 0 down to -tf

    # Reformat outputs
    t = sol.t
    PHI = sol.y.T      # shape (n_times, 42)
    
    # The state is in columns [36..41] of PHI
    x = PHI[:, 36:42]   # shape (n_times, 6)

    # The final row's first 36 columns = flattened STM at t=tf
    phi_tf_flat = PHI[-1, :36]
    phi_T = phi_tf_flat.reshape((6, 6))

    return x, t, phi_T, PHI


def monodromy_matrix(x0, mu, period, **solve_kwargs):
    """
    Compute the monodromy matrix for a periodic orbit.
    
    The monodromy matrix is the state transition matrix evaluated over one 
    orbital period of a periodic orbit. Its eigenvalues (the Floquet multipliers)
    determine the stability properties of the orbit.
    
    Parameters
    ----------
    x0 : array_like
        Initial state vector [x, y, z, vx, vy, vz] representing a point on the periodic orbit
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    period : float
        Period of the orbit
    **solve_kwargs
        Additional keyword arguments passed to compute_stm
    
    Returns
    -------
    M : ndarray
        6x6 monodromy matrix
    """
    _, _, M, _ = compute_stm(x0, mu, period, **solve_kwargs)
    return M


def stability_indices(monodromy):
    """
    Compute stability indices from the monodromy matrix eigenvalues.
    
    For a periodic orbit in the CR3BP, the stability indices are calculated from
    the non-trivial eigenvalues of the monodromy matrix. These indices help 
    characterize the stability properties of the orbit.
    
    Parameters
    ----------
    monodromy : ndarray
        6x6 monodromy matrix
    
    Returns
    -------
    nu : tuple of float
        Stability indices calculated from the eigenvalues
    eigenvalues : ndarray
        The eigenvalues of the monodromy matrix
    """
    # Calculate eigenvalues of the monodromy matrix
    eigs = np.linalg.eigvals(monodromy)
    
    # Sort eigenvalues by magnitude
    eigs = sorted(eigs, key=abs, reverse=True)
    
    # Stability indices (skip the first two which should be close to 1)
    # Use the second pair (index 2 and 3) for in-plane stability
    # Use the third pair (index 4 and 5) for out-of-plane stability
    nu1 = 0.5 * (eigs[2] + 1/eigs[2])
    nu2 = 0.5 * (eigs[4] + 1/eigs[4])
    
    return (nu1, nu2), eigs
