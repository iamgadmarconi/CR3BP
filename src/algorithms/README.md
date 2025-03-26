# Astrodynamics Algorithms

This package provides tools for analyzing and simulating dynamics in the Circular Restricted Three-Body Problem (CR3BP).

## Package Structure

The code is organized into several submodules based on functionality:

- **core**: Fundamental mathematical functions and constants
  - `lagrange_points.py`: Computation of Lagrange points
  - `energy.py`: Energy/Jacobi constant calculations

- **dynamics**: Equations of motion and numerical propagation
  - `equations.py`: Differential equations for the CR3BP
  - `propagator.py`: Numerical integration methods

- **orbits**: Computation and analysis of periodic orbits
  - `halo.py`: Halo orbit families
  - `lyapunov.py`: Lyapunov orbit families
  - `utils.py`: Orbit utility functions

- **manifolds**: Invariant manifold computation and manipulation
  - `manifold.py`: Computing stable/unstable manifolds
  - `math.py`: Mathematical tools for manifold analysis
  - `utils.py`: Utility functions for manifold processing

- **analysis**: Tools for analyzing CR3BP dynamics and stability
  - *(Future expansion)*

## Usage Examples

```python
import numpy as np
from src.algorithms import crtbp_energy, propagate_orbit, lagrange_point_locations

# Set the mass parameter (e.g., Earth-Moon)
mu = 0.012150585609624

# Get Lagrange point locations
L1, L2, L3, L4, L5 = lagrange_point_locations(mu)
print(f"L1 position: {L1}")

# Define initial state near L1
initial_state = L1 + np.array([0.001, 0, 0, 0, 0.001, 0])

# Compute energy
energy = crtbp_energy(initial_state, mu)
print(f"Energy: {energy}")

# Propagate orbit
tspan = np.linspace(0, 10, 1000)  # Time span for integration
sol = propagate_orbit(initial_state, mu, tspan)

# Plot trajectory
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.plot(sol.y[0], sol.y[1])
plt.scatter(L1[0], L1[1], marker='x', color='red', s=100, label='L1')
plt.scatter(-mu, 0, marker='o', color='blue', s=100, label='Earth')
plt.scatter(1-mu, 0, marker='o', color='gray', s=50, label='Moon')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title('Trajectory near L1')
plt.show()
``` 