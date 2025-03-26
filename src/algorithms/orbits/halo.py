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
        
    def differential_correction(self, target_state=None, tol=1e-12, max_iter=250, forward=1, **kwargs):
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
            self.initial_state, self.mu, tol=tol, max_iter=max_iter, forward=forward,
            solver_kwargs=kwargs
        )
        self.initial_state = corrected_state
        self.period = 2 * half_period
        return self.initial_state
        
    def generate_family(self, dz=1e-4, forward=1, tol=1e-12, max_iter=250, save=False, **kwargs):
        """
        Generate a family of Halo orbits by varying the z-amplitude.
        
        Parameters
        ----------
        dz : float, optional
            Step size for incrementing z-amplitude. Default is 1e-4.
        forward : {1, -1}, optional
            Direction of time integration:
            * 1: forward in time (default)
            * -1: backward in time
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
        # Get the z range limits
        z_min, z_max = _z_range(self.mu, self.L_i, self.initial_state)
        
        # Ensure dz moves in the correct direction (sign)
        if z_max < z_min and dz > 0:
            dz = -dz
            
        # Create the sequence of z values from current z to z_max
        current_z = self.initial_state[2]
        z_values = np.arange(current_z, z_max, dz)
        
        # Force at least 15 points in the sequence if there aren't enough
        if len(z_values) < 15:
            z_values = np.linspace(current_z, z_max, 15)

        family = []
        family.append(self)  # Add the current orbit as first member
        
        # Create new orbits by incrementing z
        for z_val in tqdm(z_values[1:], desc="Halo family"):
            # Create a new orbit with the next z value
            next_state = np.copy(family[-1].initial_state)
            next_state[2] = z_val
            
            orbit = HaloOrbit(self.mu, next_state, L_i=self.L_i, northern=self.northern)
            orbit.differential_correction(forward=forward, tol=tol, max_iter=max_iter, **kwargs)
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
    # Determine sign (won) and which "primary" to use
    gamma = _gamma_L(mu, L_i)
    
    if L_i == 1:
        won = +1
        primary = 1 - mu
    elif L_i == 2:
        won = -1
        primary = 1 - mu 
    elif L_i == 3:
        won = +1
        primary = -mu
    else:
        raise ValueError(f"Halo orbits only supported for L1, L2, L3 (got L{L_i})")
    
    # Set n for northern/southern family
    n = 1 if northern else -1
    
    # Coefficients c(2), c(3), c(4)
    c = [0.0, 0.0, 0.0, 0.0, 0.0]  # just to keep 5 slots: c[2], c[3], c[4]
    
    if L_i == 3:
        for N in [2, 3, 4]:
            c[N] = (1 / gamma**3) * (
                (1 - mu) + (-primary * gamma**(N + 1)) / ((1 + gamma)**(N + 1))
            )
    else:
        for N in [2, 3, 4]:
            c[N] = (1 / gamma**3) * (
                (won**N) * mu 
                + ((-1)**N)
                * (primary * gamma**(N + 1) / ((1 + (-won) * gamma)**(N + 1)))
            )

    # Solve for lambda (the in-plane frequency)
    polylambda = [
        1,
        0,
        c[2] - 2,
        0,
        - (c[2] - 1) * (1 + 2 * c[2]),
    ]
    lambda_roots = np.roots(polylambda)

    # Pick the appropriate root based on L_i
    if L_i == 3:
        lam = abs(lambda_roots[2])  # third element in 0-based indexing
    else:
        lam = abs(lambda_roots[0])  # first element in 0-based indexing

    # Calculate parameters
    k = 2 * lam / (lam**2 + 1 - c[2])
    delta = lam**2 - c[2]

    d1 = (3 * lam**2 / k) * (k * (6 * lam**2 - 1) - 2 * lam)
    d2 = (8 * lam**2 / k) * (k * (11 * lam**2 - 1) - 2 * lam)

    a21 = (3 * c[3] * (k**2 - 2)) / (4 * (1 + 2 * c[2]))
    a22 = (3 * c[3]) / (4 * (1 + 2 * c[2]))
    a23 = - (3 * c[3] * lam / (4 * k * d1)) * (
        3 * k**3 * lam - 6 * k * (k - lam) + 4
    )
    a24 = - (3 * c[3] * lam / (4 * k * d1)) * (2 + 3 * k * lam)

    b21 = - (3 * c[3] * lam / (2 * d1)) * (3 * k * lam - 4)
    b22 = (3 * c[3] * lam) / d1

    d21 = - c[3] / (2 * lam**2)

    a31 = (
        - (9 * lam / (4 * d2)) 
        * (4 * c[3] * (k * a23 - b21) + k * c[4] * (4 + k**2)) 
        + ((9 * lam**2 + 1 - c[2]) / (2 * d2)) 
        * (
            3 * c[3] * (2 * a23 - k * b21) 
            + c[4] * (2 + 3 * k**2)
        )
    )
    a32 = (
        - (1 / d2)
        * (
            (9 * lam / 4) * (4 * c[3] * (k * a24 - b22) + k * c[4]) 
            + 1.5 * (9 * lam**2 + 1 - c[2]) 
            * (c[3] * (k * b22 + d21 - 2 * a24) - c[4])
        )
    )

    b31 = (
        0.375 / d2
        * (
            8 * lam 
            * (3 * c[3] * (k * b21 - 2 * a23) - c[4] * (2 + 3 * k**2))
            + (9 * lam**2 + 1 + 2 * c[2])
            * (4 * c[3] * (k * a23 - b21) + k * c[4] * (4 + k**2))
        )
    )
    b32 = (
        (1 / d2)
        * (
            9 * lam 
            * (c[3] * (k * b22 + d21 - 2 * a24) - c[4])
            + 0.375 * (9 * lam**2 + 1 + 2 * c[2])
            * (4 * c[3] * (k * a24 - b22) + k * c[4])
        )
    )

    d31 = (3 / (64 * lam**2)) * (4 * c[3] * a24 + c[4])
    d32 = (3 / (64 * lam**2)) * (4 * c[3] * (a23 - d21) + c[4] * (4 + k**2))

    s1 = (
        1 
        / (2 * lam * (lam * (1 + k**2) - 2 * k))
        * (
            1.5 * c[3] 
            * (
                2 * a21 * (k**2 - 2) 
                - a23 * (k**2 + 2) 
                - 2 * k * b21
            )
            - 0.375 * c[4] * (3 * k**4 - 8 * k**2 + 8)
        )
    )
    s2 = (
        1 
        / (2 * lam * (lam * (1 + k**2) - 2 * k))
        * (
            1.5 * c[3] 
            * (
                2 * a22 * (k**2 - 2) 
                + a24 * (k**2 + 2) 
                + 2 * k * b22 
                + 5 * d21
            )
            + 0.375 * c[4] * (12 - k**2)
        )
    )

    a1 = -1.5 * c[3] * (2 * a21 + a23 + 5 * d21) - 0.375 * c[4] * (12 - k**2)
    a2 = 1.5 * c[3] * (a24 - 2 * a22) + 1.125 * c[4]

    l1 = a1 + 2 * lam**2 * s1
    l2 = a2 + 2 * lam**2 * s2

    deltan = -n  # matches the original code's sign usage

    # Solve for Ax from the condition ( -del - l2*Az^2 ) / l1
    Ax = np.sqrt((-delta - l2 * Az**2) / l1)

    # Evaluate the expansions at tau1 = 0
    tau1 = 0.0
    
    x = (
        a21 * Ax**2 + a22 * Az**2
        - Ax * np.cos(tau1)
        + (a23 * Ax**2 - a24 * Az**2) * np.cos(2 * tau1)
        + (a31 * Ax**3 - a32 * Ax * Az**2) * np.cos(3 * tau1)
    )
    y = (
        k * Ax * np.sin(tau1)
        + (b21 * Ax**2 - b22 * Az**2) * np.sin(2 * tau1)
        + (b31 * Ax**3 - b32 * Ax * Az**2) * np.sin(3 * tau1)
    )
    z = (
        deltan * Az * np.cos(tau1)
        + deltan * d21 * Ax * Az * (np.cos(2 * tau1) - 3)
        + deltan * (d32 * Az * Ax**2 - d31 * Az**3) * np.cos(3 * tau1)
    )

    xdot = (
        lam * Ax * np.sin(tau1)
        - 2 * lam * (a23 * Ax**2 - a24 * Az**2) * np.sin(2 * tau1)
        - 3 * lam * (a31 * Ax**3 - a32 * Ax * Az**2) * np.sin(3 * tau1)
    )
    ydot = (
        lam
        * (
            k * Ax * np.cos(tau1)
            + 2 * (b21 * Ax**2 - b22 * Az**2) * np.cos(2 * tau1)
            + 3 * (b31 * Ax**3 - b32 * Ax * Az**2) * np.cos(3 * tau1)
        )
    )
    zdot = (
        - lam * deltan * Az * np.sin(tau1)
        - 2 * lam * deltan * d21 * Ax * Az * np.sin(2 * tau1)
        - 3 * lam * deltan * (d32 * Az * Ax**2 - d31 * Az**3) * np.sin(3 * tau1)
    )

    # Scale back by gamma using original transformation
    rx = primary + gamma * (-won + x)
    ry = -gamma * y
    rz = gamma * z

    vx = gamma * xdot
    vy = gamma * ydot
    vz = gamma * zdot

    # Return the state vector
    return np.array([rx, ry, rz, vx, vy, vz], dtype=float)

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
    northern = x0i[2] > 0  # Determine family type based on z sign
    initial_orbit = HaloOrbit(mu, x0i, L_i=L_i, northern=northern)
    initial_orbit.differential_correction(tol=tol, max_iter=max_iter, forward=forward, **solver_kwargs)
    
    # Generate family
    family = initial_orbit.generate_family(
        dz=dz, forward=forward, tol=tol, max_iter=max_iter, save=save, **solver_kwargs
    )
    
    # Extract initial states and half-periods for backward compatibility
    xH = np.array([orbit.initial_state for orbit in family])
    t1H = np.array([orbit.period/2 for orbit in family])
    
    return xH, t1H

def halo_diff_correct(x0_guess, mu, forward=1, tol=1e-12, max_iter=250, solver_kwargs=None):
    """
    Diff-correction for a halo orbit in the CR3BP (CASE=1: fix z0).
    
    Parameters
    ----------
    x0_guess : array_like, shape (6,)
        Initial state guess [x0, y0, z0, vx0, vy0, vz0].
    mu : float
        Three-body mass parameter.
    forward : {1, -1}, optional
        Direction of time integration:
        * 1: forward in time (default)
        * -1: backward in time
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
        dd_vec = np.array([[DDx1], [DDz1]])  # Shape (2,1)
        phi_2 = np.array([[phi[1, 0], phi[1, 4]]])  # Shape (1,2)
        partial = dd_vec @ phi_2  # Result is (2,2)

        # Subtract the partial derivative term, scaled by 1/Dy1
        C2 = C1 - (1/Dy1) * partial
        
        # Add regularization if matrix is nearly singular
        if np.linalg.det(C2) < 1e-10:
            C2 += np.eye(2) * 1e-10

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