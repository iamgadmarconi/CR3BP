"""
Base class for periodic orbits in the Circular Restricted Three-Body Problem.

This module provides the foundation for different types of periodic orbits (such as Halo,
Lyapunov, and Vertical orbits). It defines the common interface and methods that all
orbit classes should implement, including initialization, propagation, stability analysis,
and continuation methods.
"""

import numpy as np
from abc import ABC, abstractmethod

from src.algorithms.dynamics.propagator import propagate_orbit
from src.algorithms.dynamics.stm import compute_stm, stability_indices
from src.algorithms.core.energy import crtbp_energy, energy_to_jacobi


class PeriodicOrbit(ABC):
    """
    Abstract base class for periodic orbits in the CR3BP.
    
    This class provides a common interface for all types of periodic orbits in the CR3BP,
    with methods for propagation, stability analysis, and other common functionality.
    Specific orbit types (e.g., Halo, Lyapunov) should inherit from this class and
    implement the abstract methods.
    
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
    
    def __init__(self, mu, initial_state, period=None, L_i=None):
        """
        Initialize a periodic orbit object.
        
        Parameters
        ----------
        mu : float
            Mass parameter of the CR3BP system
        initial_state : array_like
            Initial state vector [x, y, z, vx, vy, vz]
        period : float, optional
            Orbital period (if known)
        L_i : int, optional
            Libration point index (1-5) that this orbit is associated with
        """
        self.mu = mu
        self.initial_state = np.array(initial_state, dtype=np.float64)
        self.period = period
        self.L_i = L_i
        self._trajectory = None
        self._times = None
        self._stability_info = None
        
    @property
    def energy(self):
        """Compute the energy (Hamiltonian) value of the orbit."""
        return crtbp_energy(self.initial_state, self.mu)
    
    @property
    def jacobi_constant(self):
        """Compute the Jacobi constant of the orbit."""
        return energy_to_jacobi(self.energy)
    
    def propagate(self, steps=1000, rtol=1e-12, atol=1e-12, **kwargs):
        """
        Propagate the orbit for one period.
        
        Parameters
        ----------
        steps : int, optional
            Number of time steps. Default is 1000.
        rtol : float, optional
            Relative tolerance for integration. Default is 1e-12.
        atol : float, optional
            Absolute tolerance for integration. Default is 1e-12.
        **kwargs
            Additional keyword arguments passed to the integrator
            
        Returns
        -------
        tuple
            (t, trajectory) containing the time and state arrays
        """
        if self.period is None:
            raise ValueError("Period must be set before propagation")
        
        tspan = np.linspace(0, self.period, steps)
        
        sol = propagate_orbit(
            self.initial_state, self.mu, tspan, 
            rtol=rtol, atol=atol, **kwargs
        )
        
        self._trajectory = sol.y.T  # Shape (steps, 6)
        self._times = sol.t
        
        return self._times, self._trajectory
    
    def compute_stability(self, **kwargs):
        """
        Compute stability information for the orbit.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the STM computation
            
        Returns
        -------
        tuple
            (stability_indices, eigenvalues, eigenvectors) from the monodromy matrix
        """
        if self.period is None:
            raise ValueError("Period must be set before stability analysis")
        
        # Compute STM over one period
        _, _, monodromy, _ = compute_stm(
            self.initial_state, self.mu, self.period, **kwargs
        )
        
        # Analyze stability
        stability = stability_indices(monodromy)
        self._stability_info = stability
        
        return stability
    
    @property
    def is_stable(self):
        """
        Check if the orbit is linearly stable.
        
        Returns
        -------
        bool
            True if all stability indices have magnitude <= 1, False otherwise
        """
        if self._stability_info is None:
            self.compute_stability()
        
        indices = self._stability_info[0]  # nu values from stability_indices
        
        # An orbit is stable if all stability indices have magnitude <= 1
        return np.all(np.abs(indices) <= 1.0)
    
    @abstractmethod
    def generate_family(self, parameter_range, **kwargs):
        """
        Generate a family of orbits by varying a parameter.
        
        Parameters
        ----------
        parameter_range : array_like
            Range of parameter values to use for the family
        **kwargs
            Additional keyword arguments specific to the orbit type
            
        Returns
        -------
        list
            List of orbit objects representing the family
        """
        pass
    
    @abstractmethod
    def differential_correction(self, target_state, **kwargs):
        """
        Apply differential correction to improve the initial state.
        
        Parameters
        ----------
        target_state : array_like
            Target state or constraints for the correction
        **kwargs
            Additional keyword arguments specific to the correction method
            
        Returns
        -------
        ndarray
            Corrected initial state
        """
        pass
    
    @classmethod
    @abstractmethod
    def initial_guess(cls, mu, L_i, amplitude, **kwargs):
        """
        Generate an initial guess for an orbit of this type.
        
        Parameters
        ----------
        mu : float
            Mass parameter of the CR3BP system
        L_i : int
            Libration point index (1-5)
        amplitude : float
            Characteristic amplitude parameter for the orbit
        **kwargs
            Additional keyword arguments specific to the orbit type
            
        Returns
        -------
        ndarray
            Initial state vector guess [x, y, z, vx, vy, vz]
        """
        pass 