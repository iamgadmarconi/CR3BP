"""
Numerical propagation functions for CR3BP trajectories.

This module provides methods for numerical integration of orbits in the Circular
Restricted Three-Body Problem (CR3BP), including:
- Orbit propagation with various integrators
- State transition matrix computation
- Event detection for specific orbit features

The implementation supports both forward and backward propagation, with
conventions that match MATLAB-style integration for compatibility with
existing astrodynamics codes.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .equations import crtbp_accel, variational_equations


def propagate_orbit(initial_state, mu, tspan, events=None, rtol=1e-12, atol=1e-12, 
                   method='DOP853', dense_output=True, max_step=np.inf):
    """
    Propagate an orbit in the CR3BP.
    
    This function numerically integrates the equations of motion for the CR3BP
    to propagate an orbit from the given initial state over the specified time span.
    
    Parameters
    ----------
    initial_state : array_like
        Initial state vector [x, y, z, vx, vy, vz]
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    tspan : array_like
        Time span for integration [t_start, t_end] or array of specific times
    events : callable or list of callables, optional
        Events to detect during integration (see scipy.integrate.solve_ivp)
    rtol : float, optional
        Relative tolerance for the integrator
    atol : float, optional
        Absolute tolerance for the integrator
    method : str, optional
        Integration method ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA')
    dense_output : bool, optional
        Whether to compute a continuous solution
    max_step : float, optional
        Maximum allowed step size for the integrator
    
    Returns
    -------
    sol : OdeSolution
        Solution object from scipy.integrate.solve_ivp
    
    Notes
    -----
    This function uses scipy's solve_ivp under the hood, which provides
    adaptive step size integration and dense output capabilities.
    """
    # Create a wrapper for the acceleration function to match scipy's interface
    def f(t, y):
        return crtbp_accel(y, mu)
    
    # Perform the integration
    sol = solve_ivp(
        f, [tspan[0], tspan[-1]], initial_state, 
        t_eval=tspan, events=events,
        rtol=rtol, atol=atol, 
        method=method, dense_output=dense_output,
        max_step=max_step
    )
    
    return sol


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


def propagate_with_stm(initial_state, mu, tspan, initial_stm=None, events=None, 
                      rtol=1e-12, atol=1e-12, method='DOP853', dense_output=True,
                      max_step=np.inf, forward=1):
    """
    Propagate an orbit with its State Transition Matrix (STM) in the CR3BP.
    
    This function simultaneously integrates the equations of motion and the
    variational equations to compute the orbit and its State Transition Matrix,
    which relates changes in the initial state to changes in the final state.
    
    Parameters
    ----------
    initial_state : array_like
        Initial state vector [x, y, z, vx, vy, vz]
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    tspan : array_like
        Time span for integration [t_start, t_end] or array of specific times
    initial_stm : array_like, optional
        Initial STM (6x6 identity matrix by default)
    events : callable or list of callables, optional
        Events to detect during integration (see scipy.integrate.solve_ivp)
    rtol : float, optional
        Relative tolerance for the integrator
    atol : float, optional
        Absolute tolerance for the integrator
    method : str, optional
        Integration method ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA')
    dense_output : bool, optional
        Whether to compute a continuous solution
    max_step : float, optional
        Maximum allowed step size for the integrator
    forward : int, optional
        Direction of integration (1 for forward, -1 for backward)
    
    Returns
    -------
    tuple
        Tuple containing:
        - sol : OdeSolution
            Solution object with the integrated orbit
        - stm_history : ndarray
            Array of STMs at each time point, shape (n_times, 6, 6)
    
    Notes
    -----
    The STM is particularly useful for differential correction, stability analysis,
    and computing Lyapunov exponents. For large time spans, the STM can grow
    exponentially, which might lead to numerical issues.
    """
    # Initialize STM if not provided
    if initial_stm is None:
        initial_stm = np.eye(6, dtype=np.float64)
    
    # Flatten STM for integration
    initial_stm_flat = initial_stm.flatten()
    
    # Combine STM and state into a single vector
    combined_initial = np.concatenate((initial_stm_flat, initial_state))
    
    # Create a wrapper for the variational equations
    def f(t, y):
        return variational_equations(t, y, mu, forward)
    
    # Perform the integration
    sol = solve_ivp(
        f, [tspan[0], tspan[-1]], combined_initial, 
        t_eval=tspan, events=events,
        rtol=rtol, atol=atol, 
        method=method, dense_output=dense_output,
        max_step=max_step
    )
    
    # Extract the state and STM history
    y = sol.y
    stm_flat_history = y[:36, :]
    state_history = y[36:, :]
    
    # Reshape the STM history
    n_times = stm_flat_history.shape[1]
    stm_history = np.zeros((n_times, 6, 6), dtype=np.float64)
    for i in range(n_times):
        stm_history[i] = stm_flat_history[:, i].reshape((6, 6))
    
    return sol, stm_history


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