"""
Dynamical equations for the Circular Restricted Three-Body Problem (CR3BP).

This package provides functions for computing and integrating the dynamical 
equations of motion in the CR3BP, including:
- Acceleration equations
- Jacobian matrices
- State transition matrices
- Propagation methods
"""

from .equations import crtbp_accel, jacobian_crtbp, variational_equations
from .propagator import (
    propagate_orbit, 
    propagate_with_stm, 
    propagate_crtbp, 
    propagate_variational_equations
)
from .stm import compute_stm, monodromy_matrix, stability_indices

__all__ = [
    # Equations
    'crtbp_accel',
    'jacobian_crtbp',
    'variational_equations',
    
    # Propagators
    'propagate_orbit',
    'propagate_with_stm',
    'propagate_crtbp',
    'propagate_variational_equations',
    
    # STM analysis
    'compute_stm',
    'monodromy_matrix',
    'stability_indices'
] 