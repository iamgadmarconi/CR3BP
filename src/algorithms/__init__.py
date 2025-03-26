"""
Astrodynamics algorithms for the Circular Restricted Three-Body Problem (CR3BP).

This package provides tools for analyzing and simulating dynamics in the CR3BP,
organized into several submodules:

- core:      Fundamental mathematical functions and constants
- dynamics:  Equations of motion and numerical propagation
- orbits:    Computation and analysis of periodic orbits
- manifolds: Invariant manifold computation and manipulation
- analysis:  Tools for analyzing CR3BP dynamics and stability
"""

import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import commonly used functions for easier access
from .core.energy import crtbp_energy, energy_to_jacobi, jacobi_to_energy
from .core.lagrange_points import lagrange_point_locations
from .dynamics.equations import crtbp_accel
from .dynamics.propagator import (
    propagate_orbit, 
    propagate_with_stm, 
    propagate_crtbp
)
from .dynamics.stm import compute_stm, stability_indices
from .orbits import (
    PeriodicOrbit,
    halo_family,
    lyapunov_family
)
from .manifolds import (
    compute_manifold,
    surface_of_section
)

__all__ = [
    # Core math functions
    'crtbp_energy',
    'energy_to_jacobi',
    'jacobi_to_energy',
    'lagrange_point_locations',
    
    # Dynamics
    'crtbp_accel',
    'propagate_orbit',
    'propagate_with_stm',
    'propagate_crtbp',
    'compute_stm',
    'stability_indices',
    
    # Orbits
    'PeriodicOrbit',
    'halo_family',
    'lyapunov_family',
    
    # Manifolds
    'compute_manifold',
    'surface_of_section'
]