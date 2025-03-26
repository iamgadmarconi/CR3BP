"""
Invariant manifold computation and analysis for the CR3BP.

This package provides tools for computing and analyzing stable and unstable 
manifolds of periodic orbits in the Circular Restricted Three-Body Problem.
It includes functionality for:

- Computing manifold trajectories from periodic orbits
- Analyzing manifold structure using Poincaré sections
- Transforming between different coordinate frames
- Stability analysis of equilibrium points and periodic orbits
"""

import sys
sys.path.append('.')  # Adds the current directory to the Python path

import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import commonly used functions
from .manifold import compute_manifold, ManifoldResult
from .transform import libration_to_rotating, rotating_to_libration
from .analysis import (
    eigenvalue_decomposition,
    libration_stability_analysis,
    stability_indices,
    surface_of_section
)

__all__ = [
    # Manifold computation
    'compute_manifold',
    'ManifoldResult',
    
    # Coordinate transformations
    'libration_to_rotating',
    'rotating_to_libration',
    
    # Analysis tools
    'eigenvalue_decomposition',
    'libration_stability_analysis',
    'stability_indices',
    'surface_of_section',
]