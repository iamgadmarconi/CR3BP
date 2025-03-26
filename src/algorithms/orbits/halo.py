"""
This module provides functions for generating and analyzing Halo orbits in the Circular Restricted Three-Body Problem (CR3BP).

Key functionalities:
- Compute initial conditions for Halo orbits near specified libration points
- Generate families of Halo orbits by continuation in the z-amplitude
- Perform differential correction to refine initial conditions
- Compute the state transition matrix for Halo orbits
"""

import numpy as np
from tqdm import tqdm

from src.algorithms.orbits.utils import _gamma_L, _find_x_crossing, _z_range
from src.algorithms.core.lagrange_points import get_lagrange_point
from src.algorithms.dynamics.stm import compute_stm
from src.algorithms.orbits.base import PeriodicOrbit


class HaloOrbit(PeriodicOrbit):
    """
    Halo orbit implementation for the CR3BP.
    
    Halo orbits are three-dimensional periodic orbits around libration points in the CR3BP.
    This class provides methods for creating, correcting, and analyzing Halo orbits.
    
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
    northern : bool
        Whether this is a northern (True) or southern (False) family Halo orbit
    """
    
    def __init__(self, mu, initial_state, period=None, L_i=None, northern=True):
        """
        Initialize a Halo orbit object.
        
        Parameters
        ----------
        mu : float
            Mass parameter of the CR3BP system (ratio of smaller to total mass).
        initial_state : array_like
            Initial state vector [x, y, z, vx, vy, vz]
        period : float, optional
            Orbital period (if known)
        L_i : int, optional
            Libration point index (1-5) that this orbit is associated with
        northern : bool, optional
            Whether this is a northern (z>0) or southern (z<0) family Halo orbit
        """
        super().__init__(mu, initial_state, period, L_i)
        self.northern = northern
        
    def differential_correction(self, target_state=None, tol=1e-12, max_iter=250, **kwargs):
        """
        Apply differential correction to improve the initial state.
        
        Parameters
        ----------
        target_state : array_like, optional
            Target state or constraints for the correction (not used for Halo orbits)
        tol : float, optional
            Tolerance for the differential corrector. Default is 1e-12.
        max_iter : int, optional
            Maximum number of iterations for the differential corrector. Default is 250.
        **kwargs
            Additional keyword arguments passed to the underlying solver
            
        Returns
        -------
        ndarray
            Corrected initial state
        """
        corrected_state, half_period = halo_diff_correct(
            self.initial_state, self.mu, tol=tol, max_iter=max_iter, 
            solver_kwargs=kwargs
        )
        self.initial_state = corrected_state
        self.period = 2 * half_period
        return self.initial_state
        
    def generate_family(self, dz=1e-4, tol=1e-12, max_iter=250, save=False, **kwargs):
        """
        Generate a family of Halo orbits by varying the z-amplitude.
        
        Parameters
        ----------
        dz : float, optional
            Step size for incrementing z-amplitude. Default is 1e-4.
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
            List of HaloOrbit objects representing the family
        """
        parameter_range = _z_range(self.mu, self.L_i, self.initial_state)
        z_max, z_min = parameter_range
        # Ensure dz moves in the correct direction (sign)
        if z_max < z_min and dz > 0:
            dz = -dz

        family = []
        family.append(self)  # Add the current orbit as first member
        
        # Create new orbits by incrementing z
        for i, z_val in enumerate(tqdm(parameter_range[1:], desc="Halo family")):
            # Create a new orbit with incremented z
            next_state = np.copy(family[-1].initial_state)
            next_state[2] = z_val if not np.isscalar(parameter_range) else next_state[2] + dz
            
            orbit = HaloOrbit(self.mu, next_state, L_i=self.L_i, northern=self.northern)
            orbit.differential_correction(tol=tol, max_iter=max_iter, **kwargs)
            family.append(orbit)
        
        if save:
            # Extract initial states and half-periods
            states = np.array([orbit.initial_state for orbit in family])
            half_periods = np.array([orbit.period/2 for orbit in family])
            
            np.save(r"src\models\xH.npy", states)
            np.save(r"src\models\t1H.npy", half_periods)
            
        return family
        
    @classmethod
    def initial_guess(cls, mu, L_i, amplitude, northern=True, **kwargs):
        """
        Generate an initial guess for a Halo orbit.
        
        Parameters
        ----------
        mu : float
            Mass parameter of the CR3BP system
        L_i : int
            Libration point index (1-5)
        amplitude : float
            Characteristic amplitude parameter for the orbit (z-amplitude)
        northern : bool, optional
            Whether to generate a northern (z>0) or southern (z<0) family orbit
        **kwargs
            Additional keyword arguments specific to Halo orbit initialization
            
        Returns
        -------
        HaloOrbit
            A new HaloOrbit object with the initial guess
        """
        initial_state = halo_orbit_ic(mu, L_i, amplitude, northern=northern)
        return cls(mu, initial_state, L_i=L_i, northern=northern)


# Original functions maintained for backward compatibility

def halo_family(mu, L_i, x0i, dz=1e-4, forward=1, max_iter=250, tol=1e-12, save=False, **solver_kwargs):
    """
    Generate a family of Halo orbits by continuation in the z-amplitude.

    This function systematically computes a sequence of Halo orbits with
    increasing (or decreasing) out-of-plane amplitude by stepping the z-coordinate
    of the initial guess and then applying the halo differential corrector at each step.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass).
    L_i : int
        Index of the libration point (1-5).
    x0i : array_like
        Initial condition for the first orbit in the family, a 6D state vector
        [x, y, z, vx, vy, vz] in the rotating frame.
    dz : float, optional
        Step size for increasing/decreasing the z-amplitude between orbits.
        Default is 1e-3 (dimensionless units).
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
    **solver_kwargs : dict, optional
        Additional keyword arguments that will be passed along to the ODE solver
        or other internal routines in the halo_diff_correct function.
    
    Returns
    -------
    xH : ndarray
        Array of shape (N, 6) containing the initial conditions for each orbit
        in the family. Each row is a 6D state vector [x, y, z, vx, vy, vz].
    t1H : ndarray
        Array of shape (N,) containing the times to the key crossing event (often
        taken as the "half-period") for each orbit in the family.
    
    Notes
    -----
    This routine:
    1. Determines a range of z-values (z_min to z_max).
    2. Starts with the user-provided guess, corrects it via halo_diff_correct.
    3. Iterates in small z-steps, each time re-correcting to obtain the next orbit.
    4. Optionally saves the results.

    Adjust the `_z_range` helper or the stepping logic (dz, sign, etc.) to 
    generate exactly the portion of the halo family you desire.
    """
    # Create HaloOrbit object to use new OO implementation
    initial_orbit = HaloOrbit(mu, x0i, L_i=L_i)
    initial_orbit.differential_correction(tol=tol, max_iter=max_iter, **solver_kwargs)
    
    # Determine z-range
    zmin, zmax = _z_range(mu, L_i, x0i)
    # Ensure dz moves in the correct direction (sign)
    if zmax < zmin and dz > 0:
        dz = -dz
    
    # Generate z values
    z_values = np.arange(initial_orbit.initial_state[2], zmax, dz)
    
    # Generate family
    family = initial_orbit.generate_family(
        z_values, dz=dz, tol=tol, max_iter=max_iter, save=save, **solver_kwargs
    )
    
    # Extract initial states and half-periods for backward compatibility
    xH = np.array([orbit.initial_state for orbit in family])
    t1H = np.array([orbit.period/2 for orbit in family])
    
    return xH, t1H

def halo_diff_correct(x0_guess, mu, tol=1e-12, max_iter=250, solver_kwargs=None):
    """
    Diff-correction for a halo orbit in the CR3BP (CASE=1: fix z0).
    
    Parameters
    ----------
    x0_guess : array_like, shape (6,)
        Initial state guess [x0, y0, z0, vx0, vy0, vz0].
    mu : float
        Three-body mass parameter.
    tol : float, optional
        Convergence tolerance on Dx1.  (Default: 1e-12)
    max_iter : int, optional
        Maximum number of correction iterations.  (Default: 25)
    solver_kwargs : dict, optional
        Additional keyword arguments passed to the underlying ODE solver 
        in `_find_x_crossing` and `_compute_stm`.

    Returns
    -------
    XH : ndarray, shape (6,)
        Corrected state.
    TH : float
        Time at half-period crossing (when y=0 crossing is found).
    PERIOD : float
        Full estimated orbit period (2 * TH).
    """
    forward = 1
    X0 = np.copy(x0_guess)

    if solver_kwargs is None:
        solver_kwargs = {}

    # First, check if initial guess already meets the tolerance
    t1, xx1 = _find_x_crossing(X0, mu, forward=forward, **solver_kwargs)
    x1, y1, z1, Dx1, Dy1, Dz1 = xx1
    
    if abs(Dx1) <= tol and abs(Dz1) <= tol:
        # Initial guess is already perfect
        return X0, t1

    # We will iterate until Dx1 is small enough
    Dx1 = 1.0
    attempt = 0

    # For convenience:
    mu2 = 1.0 - mu

    while abs(Dx1) > tol:
        if attempt > max_iter:
            raise RuntimeError("Maximum number of correction attempts exceeded.")

        # 1) Integrate until we find the x-crossing
        t1, xx1 = _find_x_crossing(X0, mu, forward=forward, **solver_kwargs)
        x1, y1, z1, Dx1, Dy1, Dz1 = xx1

        # 2) Compute the state transition matrix at that crossing
        #    x, t, phi, PHI = _compute_stm(...)
        #      - x, t are the raw trajectory & time arrays (unused below)
        #      - phi is the final fundamental solution matrix (6x6)
        #      - PHI is the full time history if needed (not used here)
        _, _, phi, _ = compute_stm(X0, mu, t1, forward=forward, **solver_kwargs)

        # 3) Compute partial derivatives for correction
        #    (these replicate the CR3BP equations used in the Matlab code)
        rho1 = 1.0 / ((x1 + mu)**2 + y1**2 + z1**2)**1.5
        rho2 = 1.0 / ((x1 - mu2)**2 + y1**2 + z1**2)**1.5

        # second-derivatives
        omgx1 = -(mu2 * (x1 + mu) * rho1) - (mu * (x1 - mu2) * rho2) + x1
        DDz1  = -(mu2 * z1 * rho1) - (mu * z1 * rho2)
        DDx1  = 2.0 * Dy1 + omgx1

        # 4) CASE=1 Correction: fix z0
        #    We want to kill Dx1 and Dz1 by adjusting x0 and Dy0.
        #    In the Matlab code:
        #
        #    C1 = [phi(4,1) phi(4,5);
        #          phi(6,1) phi(6,5)];
        #    C2 = C1 - (1/Dy1)*[DDx1 DDz1]'*[phi(2,1) phi(2,5)];
        #    C3 = inv(C2)*[-Dx1 -Dz1]';
        #    dx0  = C3(1);
        #    dDy0 = C3(2);

        # In Python, remember phi is zero-indexed.  The original phi(i,j)
        # in Matlab is phi[i-1, j-1] in Python:
        #   phi(4,1) --> phi[3,0]
        #   phi(4,5) --> phi[3,4]
        #   phi(2,1) --> phi[1,0]   etc.
        C1 = np.array([[phi[3, 0], phi[3, 4]],
                       [phi[5, 0], phi[5, 4]]])

        # Vector for partial derivative in the (Dx, Dz) direction
        # [DDx1, DDz1]^T (2x1) times [phi(2,1), phi(2,5)] (1x2)
        partial = np.array([[DDx1], [DDz1]]) @ np.array([[phi[1, 0], phi[1, 4]]])

        # Subtract the partial derivative term, scaled by 1/Dy1
        C2 = C1 - (1/Dy1) * partial

        # Compute the correction
        C3 = np.linalg.solve(C2, np.array([[-Dx1], [-Dz1]]))

        # Apply the correction
        dx0 = C3[0, 0]
        dDy0 = C3[1, 0]

        # Update the initial guess
        X0[0] += dx0
        X0[4] += dDy0

        # Increment iteration counter
        attempt += 1

    # Return the corrected state and the half-period
    return X0, t1


def halo_orbit_ic(mu, L_i, Az=0.01, northern=True):
    """
    Generate initial conditions for a Halo orbit around a libration point.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system
    L_i : int
        Libration point index (1-5)
    Az : float, optional
        Amplitude of the Halo orbit in the z-direction. Default is 0.01.
    northern : bool, optional
        Whether to generate a northern (z>0) or southern (z<0) family orbit
    
    Returns
    -------
    ndarray
        6D state vector [x, y, z, vx, vy, vz] in the rotating frame
    """
    # Get the libration point location
    L_point = get_lagrange_point(mu, L_i)
    gamma = _gamma_L(mu, L_i)
    
    # Compute the Richardson parameters for a Halo orbit
    c2 = 1/2 * ((mu + (1-mu) * gamma**3) / gamma**3)
    c3 = -1/4 * (3*mu + 2*(1-mu)*gamma**3) / gamma**3
    c4 = -3/8 * ((3*mu + (1-mu)*gamma**3) / gamma**3)
    
    # Compute frequencies using the linearized equations
    lambda_squared = (c2 + np.sqrt(9*c2**2 - 8*c2)) / 2
    lambda_value = np.sqrt(lambda_squared)
    omega = lambda_value
    
    # Compute coefficients for the series expansion
    a21 = 3*c3*(lambda_squared - 2) / (4 * (1 + 2*c2))
    a22 = 3*c3 / (4 * (1 + 2*c2))
    a23 = -3*c3*lambda_squared / (4*lambda_squared * (1 + 2*c2))
    a24 = -3*c3*lambda_squared / (4*lambda_squared * (1 + 2*c2))
    
    a31 = -9*lambda_squared*c3**2/(4*(4*lambda_squared-c2))
    
    d21 = -6*lambda_squared*c3/(lambda_squared * (4*lambda_squared - c2))
    b21 = -3*c3*lambda_squared / (2*lambda_squared * (1 + 2*c2))
    b22 = 3*c3 / (2 * (1 + 2*c2))
    b31 = 3*c3/(2*lambda_squared*(3*lambda_squared-1))
    b32 = 3*c3/(2*lambda_squared*(3*lambda_squared-1))
    
    d1 = 3*a31 - 2*b21*a21 + b22*a22 + b31*a23 + b32*a24
    d2 = 2*a21
    
    # Convert z-amplitude to a normalized amplitude
    # Flip the sign for southern family
    Az = Az if northern else -Az
    
    # Richardson's third-order approximation
    ax = d1/d2 * Az**2
    
    # Calculate initial position and velocity
    tau1 = np.arctan(-lambda_value/3)
    tau2 = np.arctan(-lambda_value)
    
    x0 = L_point[0] + ax - a21*Az**2 - a31*Az**3
    y0 = 0
    z0 = Az
    
    vx0 = 0
    vy0 = -lambda_value*ax + d21*Az**2
    vz0 = 0
    
    return np.array([x0, y0, z0, vx0, vy0, vz0])