"""
Lagrange point model for the CR3BP.

This module defines a hierarchy of classes representing Lagrange (libration) points
in the Circular Restricted Three-Body Problem (CR3BP). The implementation provides
a clean object-oriented interface to the dynamics and stability properties of
Lagrange points, with specialized handling for collinear points (L1, L2, L3) and
triangular points (L4, L5).

The class hierarchy consists of:
- LagrangePoint (abstract base class)
- CollinearPoint (for L1, L2, L3)
- TriangularPoint (for L4, L5)
- Concrete classes for each point (L1Point, L2Point, etc.)

Each class provides methods for computing position, stability analysis, and
eigenvalue decomposition appropriate to the specific dynamics of that point type.
"""

import numpy as np
import mpmath as mp
import warnings
from abc import ABC, abstractmethod

# Import existing dynamics functionality
from src.algorithms.dynamics.equations import jacobian_crtbp
from src.algorithms.manifolds.analysis import eigenvalue_decomposition

# Set mpmath precision to 50 digits for root finding
mp.mp.dps = 50


class LagrangePoint(ABC):
    """
    Abstract base class for Lagrange points in the CR3BP.
    
    This class provides the common interface and functionality for all 
    Lagrange points. Specific point types (collinear, triangular) will
    extend this class with specialized implementations.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    point_index : int
        The Lagrange point index (1-5)
    """
    
    def __init__(self, mu, point_index):
        """Initialize a Lagrange point with the mass parameter and point index."""
        self.mu = mu
        self.point_index = point_index
        self._position = None
        self._stability_info = None
    
    @property
    def position(self):
        """
        Get the position of the Lagrange point in the rotating frame.
        
        Returns
        -------
        ndarray
            3D vector [x, y, z] representing the position
        """
        if self._position is None:
            self._position = self._calculate_position()
        return self._position
    
    def analyze_stability(self, discrete=0, delta=1e-4):
        """
        Analyze the stability properties of the Lagrange point.
        
        Parameters
        ----------
        discrete : int, optional
            Classification mode for eigenvalues:
            * 0: continuous-time system (classify by real part sign)
            * 1: discrete-time system (classify by magnitude relative to 1)
        delta : float, optional
            Tolerance for classification
            
        Returns
        -------
        tuple
            (sn, un, cn, Ws, Wu, Wc) containing:
            - sn: stable eigenvalues
            - un: unstable eigenvalues
            - cn: center eigenvalues
            - Ws: eigenvectors spanning stable subspace
            - Wu: eigenvectors spanning unstable subspace
            - Wc: eigenvectors spanning center subspace
        """
        if self._stability_info is None:
            # Compute the system Jacobian at the Lagrange point
            pos = self.position
            A = jacobian_crtbp(pos[0], pos[1], pos[2], self.mu)
            
            # Perform eigenvalue decomposition and classification
            self._stability_info = eigenvalue_decomposition(A, discrete, delta)
        
        return self._stability_info
    
    @property
    def eigenvalues(self):
        """
        Get the eigenvalues of the linearized system at the Lagrange point.
        
        Returns
        -------
        tuple
            (stable_eigenvalues, unstable_eigenvalues, center_eigenvalues)
        """
        sn, un, cn, _, _, _ = self.analyze_stability()
        return sn, un, cn
    
    @property
    def eigenvectors(self):
        """
        Get the eigenvectors of the linearized system at the Lagrange point.
        
        Returns
        -------
        tuple
            (stable_eigenvectors, unstable_eigenvectors, center_eigenvectors)
        """
        _, _, _, Ws, Wu, Wc = self.analyze_stability()
        return Ws, Wu, Wc
    
    @abstractmethod
    def _calculate_position(self):
        """
        Calculate the position of the Lagrange point.
        
        This is an abstract method that must be implemented by subclasses.
        
        Returns
        -------
        ndarray
            3D vector [x, y, z] representing the position
        """
        pass


class CollinearPoint(LagrangePoint):
    """
    Base class for collinear Lagrange points (L1, L2, L3).
    
    The collinear points lie on the x-axis connecting the two primary
    bodies. They are characterized by having unstable dynamics with
    saddle-center stability (one unstable direction, two center directions).
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    point_index : int
        The Lagrange point index (must be 1, 2, or 3)
    """
    
    def __init__(self, mu, point_index):
        """Initialize a collinear Lagrange point."""
        if point_index not in [1, 2, 3]:
            raise ValueError(f"Collinear point index must be 1, 2, or 3, not {point_index}")
        super().__init__(mu, point_index)
    
    def _dOmega_dx(self, x):
        """
        Compute the derivative of the effective potential with respect to x.
        
        Parameters
        ----------
        x : float
            x-coordinate in the rotating frame
        
        Returns
        -------
        float
            Value of dΩ/dx at the given x-coordinate
        """
        mu = self.mu
        r1 = abs(x + mu)
        r2 = abs(x - (1 - mu))
        return x - (1 - mu) * (x + mu) / (r1**3) - mu * (x - (1 - mu)) / (r2**3)


class L1Point(CollinearPoint):
    """
    L1 Lagrange point, located between the two primary bodies.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, mu):
        """Initialize the L1 Lagrange point."""
        super().__init__(mu, 1)
    
    def _calculate_position(self):
        """
        Calculate the position of the L1 point.
        
        Returns
        -------
        ndarray
            3D vector [x, 0, 0] giving the position of L1
        """
        x = mp.findroot(lambda x: self._dOmega_dx(x), [-self.mu + 0.01, 1 - self.mu - 0.01])
        x = float(x)
        return np.array([x, 0, 0], dtype=np.float64)


class L2Point(CollinearPoint):
    """
    L2 Lagrange point, located beyond the smaller primary body.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, mu):
        """Initialize the L2 Lagrange point."""
        super().__init__(mu, 2)
    
    def _calculate_position(self):
        """
        Calculate the position of the L2 point.
        
        Returns
        -------
        ndarray
            3D vector [x, 0, 0] giving the position of L2
        """
        x = mp.findroot(lambda x: self._dOmega_dx(x), [1.0, 2.0])
        x = float(x)
        return np.array([x, 0, 0], dtype=np.float64)


class L3Point(CollinearPoint):
    """
    L3 Lagrange point, located beyond the larger primary body.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, mu):
        """Initialize the L3 Lagrange point."""
        super().__init__(mu, 3)
    
    def _calculate_position(self):
        """
        Calculate the position of the L3 point.
        
        Returns
        -------
        ndarray
            3D vector [x, 0, 0] giving the position of L3
        """
        x = mp.findroot(lambda x: self._dOmega_dx(x), [-self.mu - 0.01, -2.0])
        x = float(x)
        return np.array([x, 0, 0], dtype=np.float64)


class TriangularPoint(LagrangePoint):
    """
    Base class for triangular Lagrange points (L4, L5).
    
    The triangular points form equilateral triangles with the two primary
    bodies. They are characterized by having center stability (stable)
    for mass ratios μ < 0.0385, and unstable for larger mass ratios.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    point_index : int
        The Lagrange point index (must be 4 or 5)
    """
    
    def __init__(self, mu, point_index):
        """Initialize a triangular Lagrange point."""
        if point_index not in [4, 5]:
            raise ValueError(f"Triangular point index must be 4 or 5, not {point_index}")
        super().__init__(mu, point_index)
        
        # Check stability based on mass ratio
        if mu > 0.0385:
            warnings.warn(f"Triangular points are unstable for mu > 0.0385 (current mu = {mu})")


class L4Point(TriangularPoint):
    """
    L4 Lagrange point, forming an equilateral triangle with the two primary bodies,
    located above the x-axis (positive y).
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, mu):
        """Initialize the L4 Lagrange point."""
        super().__init__(mu, 4)
    
    def _calculate_position(self):
        """
        Calculate the position of the L4 point.
        
        Returns
        -------
        ndarray
            3D vector [x, y, 0] giving the position of L4
        """
        x = 1 / 2 - self.mu
        y = np.sqrt(3) / 2
        return np.array([x, y, 0], dtype=np.float64)


class L5Point(TriangularPoint):
    """
    L5 Lagrange point, forming an equilateral triangle with the two primary bodies,
    located below the x-axis (negative y).
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, mu):
        """Initialize the L5 Lagrange point."""
        super().__init__(mu, 5)
    
    def _calculate_position(self):
        """
        Calculate the position of the L5 point.
        
        Returns
        -------
        ndarray
            3D vector [x, y, 0] giving the position of L5
        """
        x = 1 / 2 - self.mu
        y = -np.sqrt(3) / 2
        return np.array([x, y, 0], dtype=np.float64)


# Factory function to create Lagrange points
def create_lagrange_point(mu, point_index):
    """
    Create a specific Lagrange point object by index.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    point_index : int
        The Lagrange point index (1-5)
        
    Returns
    -------
    LagrangePoint
        An instance of the appropriate Lagrange point class
        
    Raises
    ------
    ValueError
        If an invalid point index is provided
    """
    if point_index == 1:
        return L1Point(mu)
    elif point_index == 2:
        return L2Point(mu)
    elif point_index == 3:
        return L3Point(mu)
    elif point_index == 4:
        return L4Point(mu)
    elif point_index == 5:
        return L5Point(mu)
    else:
        raise ValueError(f"Invalid Lagrange point index: {point_index}. Must be 1-5.")


# Compatibility functions to maintain backward compatibility with existing code
def get_lagrange_point(mu, point_index):
    """
    Get the position of a specific Lagrange point (compatibility function).
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    point_index : int
        Lagrange point index (1-5)
        
    Returns
    -------
    ndarray
        3D vector [x, y, z] giving the position of the specified Lagrange point
    """
    return create_lagrange_point(mu, point_index).position


def lagrange_point_locations(mu):
    """
    Compute all five libration points in the CR3BP (compatibility function).
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    tuple
        A tuple containing the positions of L1, L2, L3, L4, and L5 as ndarrays
    """
    return (get_lagrange_point(mu, 1),
            get_lagrange_point(mu, 2),
            get_lagrange_point(mu, 3),
            get_lagrange_point(mu, 4),
            get_lagrange_point(mu, 5)) 