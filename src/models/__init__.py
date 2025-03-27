import sys
sys.path.append('.')  # Adds the current directory to the Python path

import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import models
from .body import Body
from .lagrange_point import (
    LagrangePoint,
    CollinearPoint, 
    TriangularPoint,
    L1Point, 
    L2Point, 
    L3Point, 
    L4Point, 
    L5Point,
    create_lagrange_point,
    get_lagrange_point,
    lagrange_point_locations
)

# Export all model classes
__all__ = [
    'Body',
    'LagrangePoint',
    'CollinearPoint',
    'TriangularPoint',
    'L1Point',
    'L2Point',
    'L3Point',
    'L4Point',
    'L5Point',
    'create_lagrange_point',
    'get_lagrange_point',
    'lagrange_point_locations'
]