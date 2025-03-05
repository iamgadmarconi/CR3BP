"""
Numerical integration tools for the Circular Restricted Three-Body Problem (CR3BP).

This module provides numerical propagation functions for integrating both the 
standard CR3BP equations of motion and the variational equations. It serves as 
a bridge between the dynamical equations defined in dynamics.py and numerical 
integration methods from SciPy.

The propagator offers:
1. Integration of CR3BP state vectors in forward or backward time
2. Integration of combined state and state transition matrices (STM)
3. Support for variable step-size integration with customizable tolerances
4. Compatibility with MATLAB-style integration conventions

These functions are essential for trajectory design, stability analysis, and
differential correction in the CR3BP.
"""

import numba
import numpy as np
from scipy.integrate import solve_ivp

from src.dynamics.dynamics import crtbp_accel, variational_equations


def propagate_crtbp(state0, t0, tf, mu, forward=1, steps=1000, **solve_kwargs):
    """
    Propagate a state in the CR3BP from initial to final time.
    
    This function numerically integrates the CR3BP equations of motion,
    providing a trajectory from an initial state at time t0 to a final
    time tf. It handles both forward and backward propagation and is
    designed to replicate MATLAB-style integration conventions.
    
    Parameters
    ----------
    state0 : array_like
        Initial state vector [x, y, z, vx, vy, vz]
    t0 : float
        Initial time
    tf : float
        Final time
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    forward : int, optional
        Direction of integration (1 for forward, -1 for backward). Default is 1.
    steps : int, optional
        Number of time steps for output. Default is 1000.
    **solve_kwargs
        Additional keyword arguments passed to scipy.integrate.solve_ivp
    
    Returns
    -------
    sol : OdeResult
        Solution object from scipy.integrate.solve_ivp containing:
        - t: array of time points
        - y: array of solution values (shape: (6, len(t)))
        - Additional integration metadata
    
    Notes
    -----
    This function adopts MATLAB's 'FORWARD' convention where:
    1. Integration always occurs over a positive time span [|t0|, |tf|]
    2. The derivative is multiplied by 'forward' to control direction
    3. The output time array is scaled by 'forward' to reflect actual times
    
    Default integration tolerances are set to high precision (rtol=3e-14, 
    atol=1e-14) but can be overridden through solve_kwargs.
    """
    # 1) Always make the integration span positive, even if tf is negative
    t0 = abs(t0)
    tf = abs(tf)
    t_eval = np.linspace(t0, tf, steps)
    
    # 2) ODE function includes the forward sign, exactly like 'xdot = FORWARD * xdot'
    def ode_func(t, y):
        return forward * crtbp_accel(y, mu)

    # 4) Default tolerances, or user can override via solve_kwargs
    if 'rtol' not in solve_kwargs:
        solve_kwargs['rtol'] = 3e-14
    if 'atol' not in solve_kwargs:
        solve_kwargs['atol'] = 1e-14

    # 5) Integrate
    sol = solve_ivp(
        ode_func,
        [t0, tf],
        state0,
        t_eval=t_eval,
        **solve_kwargs
    )

    # 6) Finally, flip the reported times so that if forward = -1,
    #    the time array goes from 0 down to -T (like MATLAB's t=FORWARD*t)
    sol.t = forward * sol.t

    return sol

def propagate_variational_equations(PHI_init, mu, T, forward=1, steps=1000, **solve_kwargs):
    """
    Integrate the variational equations of the CR3BP.
    
    This function numerically integrates the combined system of CR3BP
    equations of motion and the state transition matrix (STM). It allows
    tracking how small perturbations in initial conditions propagate
    over time, which is essential for differential correction and 
    stability analysis.
    
    Parameters
    ----------
    PHI_init : array_like
        42-element initial vector containing:
        - First 36 elements: flattened 6x6 state transition matrix (STM)
        - Last 6 elements: state vector [x, y, z, vx, vy, vz]
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    T : float
        Final integration time (must be positive)
    forward : int, optional
        Direction of integration (1 for forward, -1 for backward). Default is 1.
    steps : int, optional
        Number of time steps for output. Default is 1000.
    **solve_kwargs
        Additional keyword arguments passed to scipy.integrate.solve_ivp
    
    Returns
    -------
    sol : OdeResult
        Solution object from scipy.integrate.solve_ivp containing:
        - t: array of time points
        - y: array of solution values (shape: (42, len(t)))
          where y[:36] is the flattened STM and y[36:42] is the state vector
        - Additional integration metadata
    
    Notes
    -----
    The state transition matrix Φ(t) maps perturbations in initial conditions
    to perturbations at time t according to δx(t) = Φ(t)·δx(0). The matrix
    is initialized as the identity matrix at time t=0.
    
    When forward=-1, the output time array is negated to represent backward
    integration, matching MATLAB's convention.
    """
    def rhs(t, PHI):
        return variational_equations(t, PHI, mu, forward)

    t_span = (0.0, T)
    t_eval = np.linspace(0.0, T, steps)

    sol = solve_ivp(rhs, t_span, PHI_init, t_eval=t_eval, 
                    **solve_kwargs)

    # If you want the time array to reflect backward integration for forward=-1:
    if forward == -1:
        sol.t = -sol.t  # times run 0 -> -T, matching MATLAB's "t=FORWARD*t" logic

    return sol
