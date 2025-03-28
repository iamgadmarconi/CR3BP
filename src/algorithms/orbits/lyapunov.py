"""
Lyapunov orbit computation module for the Circular Restricted Three-Body Problem (CR3BP).

This module provides functions to compute Lyapunov orbits around libration points
in the CR3BP. Lyapunov orbits are planar periodic orbits that exist in the vicinity
of the collinear libration points (L1, L2, L3). The module includes functionality to:

1. Generate initial conditions for Lyapunov orbits
2. Apply differential correction to refine these initial conditions
3. Compute entire families of Lyapunov orbits of increasing amplitude

The implementation uses linearized dynamics around libration points to generate
initial guesses, and then applies numerical techniques to correct these guesses
into precise periodic solutions.
"""

import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp

from src.algorithms.dynamics.propagator import propagate_crtbp
from src.algorithms.dynamics.equations import variational_equations
from src.algorithms.orbits.utils import _find_x_crossing, _x_range
from src.algorithms.manifolds.math import _libration_frame_eigenvectors
from src.algorithms.manifolds.analysis import libration_stability_analysis
from src.algorithms.core.lagrange_points import get_lagrange_point
from src.algorithms.orbits.base import PeriodicOrbit


class LyapunovOrbit(PeriodicOrbit):
    """
    Lyapunov orbit implementation for the CR3BP.
    
    Lyapunov orbits are planar periodic orbits that exist in the vicinity of the
    collinear libration points (L1, L2, L3). This class provides methods for
    computing, analyzing, and continuing families of Lyapunov orbits.
    
    Attributes
    ----------
    mu : float
        Mass parameter of the CR3BP system
    initial_state : ndarray
        Initial state vector [x, y, z, vx, vy, vz]
    period : float
        Orbital period
    L_i : int
        Libration point index (1-5)
    """
    
    def differential_correction(self, target_state=None, tol=1e-12, max_iter=250, forward=1, **kwargs):
        """
        Apply differential correction to improve the initial state.
        
        This method adjusts the initial velocity component vy0 so that the
        resulting trajectory forms a periodic Lyapunov orbit.
        
        Parameters
        ----------
        target_state : array_like, optional
            Target state or constraints for the correction (not used for Lyapunov orbits)
        tol : float, optional
            Tolerance for the differential corrector. Default is 1e-12.
        max_iter : int, optional
            Maximum number of iterations for the differential corrector. Default is 250.
        forward : {1, -1}, optional
            Direction of time integration:
            * 1: forward in time (default)
            * -1: backward in time
        **kwargs
            Additional keyword arguments passed to the underlying solver
            
        Returns
        -------
        ndarray
            Corrected initial state
        """
        corrected_state, half_period = lyapunov_diff_correct(
            self.initial_state, self.mu, forward=forward, 
            tol=tol, max_iter=max_iter, **kwargs
        )
        self.initial_state = corrected_state
        self.period = 2 * half_period
        return self.initial_state
        
    def generate_family(self, dx=1e-4, forward=1, tol=1e-12, max_iter=250, save=False, **kwargs):
        """
        Generate a family of Lyapunov orbits by varying the x-amplitude.
        
        Parameters
        ----------
        dx : float, optional
            Step size for incrementing x-amplitude. Default is 1e-4.
        forward : {1, -1}, optional
            Direction of time integration:
            * 1: forward in time (default)
            * -1: backward in time
        tol : float, optional
            Tolerance for the differential corrector. Default is 1e-12.
        max_iter : int, optional
            Maximum number of iterations for the differential corrector. Default is 250.
        save : bool, optional
            Whether to save the computed family to disk. Default is False.
        **kwargs
            Additional keyword arguments passed to the underlying solver
            
        Returns
        -------
        list
            List of LyapunovOrbit objects representing the family
        """
        # Determine the range of x values
        xmin, xmax = _x_range(self.mu, self.L_i, self.initial_state)
        n = int(np.floor((xmax - xmin)/dx + 1))
        
        family = []
        family.append(self)  # Add the current orbit as first member
        
        # Step through the range with a progress bar
        for j in tqdm(range(1, n), desc="Lyapunov family"):
            # Create a new orbit with incremented x
            next_state = np.copy(family[-1].initial_state)
            next_state[0] += dx  # increment the x0 amplitude
            
            orbit = LyapunovOrbit(self.mu, next_state, L_i=self.L_i)
            orbit.differential_correction(forward=forward, tol=tol, max_iter=max_iter, **kwargs)
            family.append(orbit)
        
        if save:
            # Extract initial states and half-periods
            states = np.array([orbit.initial_state for orbit in family])
            half_periods = np.array([orbit.period/2 for orbit in family])
            
            np.save(r"src\models\xL.npy", states)
            np.save(r"src\models\t1L.npy", half_periods)
            
        return family
        
    @classmethod
    def initial_guess(cls, mu, L_i, amplitude, **kwargs):
        """
        Generate an initial guess for a Lyapunov orbit.
        
        Parameters
        ----------
        mu : float
            Mass parameter of the CR3BP system
        L_i : int
            Libration point index (1-5)
        amplitude : float
            Characteristic amplitude parameter for the orbit (x-amplitude)
        **kwargs
            Additional keyword arguments specific to Lyapunov orbit initialization
            
        Returns
        -------
        LyapunovOrbit
            A new LyapunovOrbit object with the initial guess
        """
        initial_state = lyapunov_orbit_ic(mu, L_i, Ax=amplitude)
        return cls(mu, initial_state, L_i=L_i)


# Original functions maintained for backward compatibility

def lyapunov_orbit_ic(mu, L_i, Ax=1e-5):
    """
    Generate initial conditions for a Lyapunov orbit near a libration point.
    
    This function uses the linearized dynamics around a libration point to generate
    an initial guess for a Lyapunov orbit. It computes the eigenvectors of the 
    linearized system and uses them to displace the initial state from the 
    libration point.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    L_i : int
        Index of the libration point (1-5)
    Ax : float, optional
        Amplitude of the orbit in the x-direction (dimensionless units).
        Default is 1e-5, which produces a very small orbit.
    
    Returns
    -------
    ndarray
        A 6D state vector [x, y, z, vx, vy, vz] representing the initial 
        condition for a Lyapunov orbit in the rotating frame
    
    Notes
    -----
    The initial condition is constructed to be at y=0 with a specific
    displacement in the x-direction. This is a common starting point
    for Lyapunov orbits, which are symmetric about the x-axis.
    
    For small amplitudes, this initial guess is very accurate due to the
    validity of the linearized dynamics. For larger amplitudes, further
    differential correction may be necessary.
    """
    u1, u2, u, v = _libration_frame_eigenvectors(mu, L_i, orbit_type="lyapunov")

    L_i = get_lagrange_point(mu, L_i)

    displacement = Ax * u

    x_L_i = L_i[0]

    state = np.array([x_L_i, 0, 0, 0], dtype=np.float64) + displacement

    x, y, vx, vy = state[0], state[1], state[2], state[3]

    return np.array([x, y, 0, vx, vy, 0], dtype=np.float64)


def lyapunov_family(mu, L_i, x0i, dx=1e-4, forward=1, max_iter=250, tol=1e-12, save=False, **solver_kwargs):
    """
    Generate a family of Lyapunov orbits by continuation in the x-amplitude.
    
    This function systematically computes a sequence of Lyapunov orbits with
    increasing amplitude by stepping the x-coordinate and applying differential
    correction at each step. It creates a family of orbits ranging from small
    (nearly linear) to large (highly nonlinear) amplitudes.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    L_i : int
        Index of the libration point (1-5)
    x0i : array_like
        Initial condition for the first orbit in the family, a 6D state vector
        [x, y, z, vx, vy, vz] in the rotating frame
    dx : float, optional
        Step size for increasing the x-amplitude between orbits.
        Default is 0.0001 (dimensionless units).
    forward : {1, -1}, optional
        Direction of time integration:
        * 1: forward in time (default)
        * -1: backward in time
    max_iter : int, optional
        Maximum number of iterations for the differential corrector.
        Default is 250.
    tol : float, optional
        Tolerance for the differential corrector. Default is 1e-12.
    save : bool, optional
        Whether to save the computed family to disk. Default is False.
    
    Returns
    -------
    xL : ndarray
        Array of shape (N, 6) containing the initial conditions for each orbit
        in the family. Each row is a 6D state vector [x, y, z, vx, vy, vz].
    t1L : ndarray
        Array of shape (N,) containing the half-periods for each orbit in the family.
    
    Notes
    -----
    The function works by computing a range of x-values based on the libration
    point position and input initial condition, then stepping through this range
    with the specified step size. At each step, it applies differential correction
    to find a precise Lyapunov orbit.
    
    The computation uses the previous orbit's initial condition as the starting
    point for the next orbit, incrementing the x-coordinate by dx each time.
    This continuation approach allows efficient computation of the entire family.
    
    If save=True, the results are saved to disk in the src/models directory.
    """
    # Create LyapunovOrbit object to use new OO implementation
    initial_orbit = LyapunovOrbit(mu, x0i, L_i=L_i)
    initial_orbit.differential_correction(forward=forward, tol=tol, max_iter=max_iter, **solver_kwargs)
    
    # Generate the family using the class method
    family = initial_orbit.generate_family(
        dx=dx, forward=forward, tol=tol, max_iter=max_iter, save=save, **solver_kwargs
    )
    
    # Extract initial states and half-periods for backward compatibility
    xL = np.array([orbit.initial_state for orbit in family])
    t1L = np.array([orbit.period/2 for orbit in family])
    
    return xL, t1L

def lyapunov_diff_correct(x0_guess, mu, forward=1, tol=1e-12, max_iter=250, **solver_kwargs):
    """
    Apply differential correction to refine an initial condition for a Lyapunov orbit.
    
    This function uses a numerical differential correction technique to adjust
    the initial velocity component vy0 of a state vector so that the resulting
    trajectory forms a periodic orbit. Specifically, it ensures that when the
    trajectory crosses the x-axis (y=0), the x-velocity component (vx) is zero,
    which is a necessary condition for a symmetric periodic orbit.
    
    Parameters
    ----------
    x0_guess : array_like
        Initial guess for the orbit, a 6D state vector [x, y, z, vx, vy, vz]
        in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    forward : {1, -1}, optional
        Direction of time integration:
        * 1: forward in time (default)
        * -1: backward in time
    tol : float, optional
        Convergence tolerance for the differential corrector. Default is 1e-12.
    max_iter : int, optional
        Maximum number of iterations for the differential corrector. 
        Default is 250.
    
    Returns
    -------
    x0 : ndarray
        Corrected initial condition that produces a Lyapunov orbit
    half_period : float
        Half-period of the resulting orbit (time to reach the opposite x-axis crossing)
    
    Raises
    ------
    RuntimeError
        If the differential corrector does not converge within max_iter iterations
    
    Notes
    -----
    The differential correction process:
    1. Propagates the trajectory until it crosses the y=0 plane
    2. Checks if vx=0 at the crossing (periodicity condition)
    3. If not, computes the partial derivative dvx/dvy0 using the STM
    4. Adjusts vy0 to make vx closer to zero at the crossing
    5. Repeats until convergence
    
    This method preserves the values of x0, y0, z0, vx0, and vz0, adjusting
    only vy0. It's designed for symmetric Lyapunov orbits where the initial
    condition is on the x-axis (y=0) and the orbit has reflective symmetry.
    """
    x0 = np.copy(x0_guess)
    attempt = 0
    
    # First, check if initial guess already meets the tolerance
    t_cross, X_cross = _find_x_crossing(x0, mu, forward=forward, **solver_kwargs)
    vx_cross = X_cross[3]
    
    if abs(vx_cross) < tol:
        # Initial guess is already perfect
        half_period = t_cross
        return x0, half_period
    
    while True:
        attempt += 1
        if attempt > max_iter:
            raise RuntimeError("Max attempts exceeded in differential corrector.")

        t_cross, X_cross = _find_x_crossing(x0, mu, forward=forward, **solver_kwargs)
        
        # The crossing states
        x_cross = X_cross[0]
        y_cross = X_cross[1]
        z_cross = X_cross[2]
        vx_cross = X_cross[3]
        vy_cross = X_cross[4]
        vz_cross = X_cross[5]
        
        if abs(vx_cross) < tol:
            # Done
            half_period = t_cross
            return x0, half_period
            
        # Build the initial condition vector for the combined state and STM.
        # New convention:
        #   First 36 elements: flattened 6x6 STM (initialized to identity)
        #   Last 6 elements: the state vector x0
        PHI0 = np.zeros(42)
        PHI0[:36] = np.eye(6).flatten()
        PHI0[36:] = x0
        
        def ode_fun(t, y):
            """
            ODE function that computes the combined state and STM derivatives.
            
            This function is passed to the ODE solver to integrate both the state 
            and the state transition matrix simultaneously.
            
            Parameters
            ----------
            t : float
                Current time
            y : ndarray
                Combined state+STM vector of length 42
            
            Returns
            -------
            ndarray
                Vector of derivatives, also of length 42
            """
            return variational_equations(t, y, mu, forward=forward)
        
        # We'll integrate to the crossing time directly
        sol = solve_ivp(ode_fun, [0, t_cross], PHI0, rtol=1e-12, atol=1e-12, dense_output=True)
        
        # Evaluate the final 42-vector at t_cross
        PHI_vec_final = sol.sol(t_cross)
        
        # Extract the STM and the final state using the new convention.
        phi_final = PHI_vec_final[:36].reshape((6, 6))
        state_final = PHI_vec_final[36:]
        
        # We want the partial derivative of vx_cross with respect to the initial vy.
        # This is given by phi_final[3,4] in 0-based indexing.
        dvx_dvy0 = phi_final[3, 4]
        
        # Basic linear correction:  dvy = - vx_cross / dvx_dvy0
        dvy = -vx_cross / dvx_dvy0
        
        # Update x0 (adjust only the vy component)
        x0[4] += dvy