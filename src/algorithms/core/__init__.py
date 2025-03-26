"""
Core mathematical functions for the Circular Restricted Three-Body Problem (CR3BP).

This package contains the fundamental mathematical tools and constants used
throughout the astrodynamics simulations.
"""

from .lagrange_points import lagrange_point_locations, get_lagrange_point
from .energy import crtbp_energy, hill_region, energy_to_jacobi, jacobi_to_energy, compute_energy_bounds

__all__ = [
    'lagrange_point_locations',
    'get_lagrange_point',
    'crtbp_energy',
    'hill_region',
    'energy_to_jacobi',
    'jacobi_to_energy',
    'compute_energy_bounds'
] 