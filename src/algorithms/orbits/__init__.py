"""
Periodic orbit computation for the Circular Restricted Three-Body Problem (CR3BP).

This package provides tools for computing and analyzing different types of periodic
orbits in the CR3BP, including:

- Halo orbits around libration points
- Lyapunov planar periodic orbits
- Tools for differential correction and orbit continuation
- Orbit family generation and classification
"""

import sys
sys.path.append('.')  # Adds the current directory to the Python path

import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import base class
from .base import PeriodicOrbit

# Import specific orbit type functions
from .halo import halo_family, halo_diff_correct, halo_orbit_ic
from .lyapunov import lyapunov_family, lyapunov_orbit_ic, lyapunov_diff_correct

__all__ = [
    # Base class
    'PeriodicOrbit',
    
    # Halo orbits
    'halo_family',
    'halo_diff_correct',
    'halo_orbit_ic',
    
    # Lyapunov orbits
    'lyapunov_family',
    'lyapunov_orbit_ic',
    'lyapunov_diff_correct'
]