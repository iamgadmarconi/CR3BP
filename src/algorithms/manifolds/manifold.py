"""
Manifold computation module for the Circular Restricted Three-Body Problem (CR3BP).

This module provides functions to compute and analyze stable and unstable manifolds
of periodic orbits in the CR3BP. It implements techniques to find initial conditions
on manifolds, propagate trajectories along these manifolds, and analyze the results
using Poincaré sections.

The module relies on state transition matrix (STM) computation to find the stable
and unstable directions in phase space, and uses eigenvalue decomposition to identify
these directions accurately.
"""

import logging
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass

from src.dynamics.propagator import propagate_crtbp
from src.dynamics.stm import _compute_stm
from src.dynamics.manifolds.math import _surface_of_section, _eig_decomp
from src.dynamics.manifolds.utils import _totime

# Get logger for this module
logger = logging.getLogger(__name__)


@dataclass
class ManifoldResult:
    """Container for manifold computation results."""
    ysos: List[float]
    ydsos: List[float]
    xW_list: List[np.ndarray]
    tW_list: List[np.ndarray]
    success_count: int
    attempt_count: int
    
    @property
    def success_rate(self) -> float:
        """Return the success rate of manifold computations."""
        return self.success_count / max(1, self.attempt_count)


def compute_manifold(x0: np.ndarray, T: float, mu: float, stbl: int = 1, 
                     direction: int = 1, forward: int = 1, step: float = 0.02, 
                     steps: int = 5000, integration_fraction: float = 0.7, 
                     show_progress: bool = True, **solver_kwargs) -> Union[Tuple, ManifoldResult]:
    """
    Computes the stable or unstable manifold of a periodic orbit in the CR3BP.
    
    This function systematically samples points along a periodic orbit, computes the
    local stable/unstable directions at each point, and propagates trajectories
    along these directions. The results are analyzed through Poincaré sections.
    
    Parameters
    ----------
    x0 : array_like
        Initial condition of the periodic orbit, a 6D state vector [x, y, z, vx, vy, vz].
    T : float
        Period of the periodic orbit in non-dimensional time units.
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass).
    stbl : {1, -1}, optional
        Manifold type selector:
        * 1: stable manifold (default)
        * -1: unstable manifold
    direction : {1, -1}, optional
        Branch selector:
        * 1: "positive" branch of the manifold (default)
        * -1: "negative" branch of the manifold
    forward : {1, -1}, optional
        Integration direction selector:
        * 1: forward in time (default)
        * -1: backward in time
    step : float, optional
        Sampling step size along the orbit (fraction of orbit from 0 to 1).
        Default is 0.02 (50 points around the orbit).
    steps : int, optional
        Number of integration steps for propagating manifold trajectories.
        Default is 5000.
    integration_fraction : float, optional
        Fraction of the orbit to integrate.
        Default is 0.7 for Lyapunov orbits and 0.8 for halo orbits.
    show_progress : bool, optional
        Whether to display a progress bar during computation.
        Default is True.
    **solver_kwargs
        Additional keyword arguments passed to the numerical integrator.
        
    Returns
    -------
    ManifoldResult
        A dataclass containing:
        - ysos: List of y-coordinates from the Poincaré section
        - ydsos: List of vy-coordinates from the Poincaré section
        - xW_list: List of propagated state trajectories
        - tW_list: List of time vectors corresponding to each trajectory
        - success_count: Number of successful manifold point computations
        - attempt_count: Total number of attempted manifold point computations
    
    Raises
    ------
    ValueError
        If input parameters are outside valid ranges.
    RuntimeError
        If manifold computation fails due to numerical issues.
    
    Notes
    -----
    This function samples the orbit at equal fractions, computes the stable/unstable
    direction at each sample point, slightly displaces the state in that direction,
    and then propagates the resulting trajectory. The results are analyzed using a
    Poincaré section with respect to the smaller primary (M=2) to observe the
    manifold structure.
    """
    # Input validation
    if not isinstance(x0, np.ndarray) or x0.size != 6:
        raise ValueError(f"x0 must be a 6-element numpy array, got {type(x0)} with size {x0.size if hasattr(x0, 'size') else 'unknown'}")
    
    if not (0 < step < 1):
        raise ValueError(f"Step size must be between 0 and 1, got {step}")
    
    if not (0 < integration_fraction <= 1):
        raise ValueError(f"Integration fraction must be between 0 and 1, got {integration_fraction}")
    
    if stbl not in [1, -1]:
        raise ValueError(f"stbl must be 1 (stable) or -1 (unstable), got {stbl}")
    
    if direction not in [1, -1]:
        raise ValueError(f"direction must be 1 (positive) or -1 (negative), got {direction}")
    
    if forward not in [1, -1]:
        raise ValueError(f"forward must be 1 (forward) or -1 (backward), got {forward}")
    
    # Ensure tolerances are set, use defaults if not provided
    solver_kwargs.setdefault('rtol', 1e-12)
    solver_kwargs.setdefault('atol', 1e-12)
    
    logger.info(f"Computing {'stable' if stbl == 1 else 'unstable'} manifold with "
                f"{'positive' if direction == 1 else 'negative'} branch, "
                f"{'forward' if forward == 1 else 'backward'} integration")
    
    # Pre-allocate data structures
    ysos = []
    ydsos = []
    xW_list = []
    tW_list = []
    success_count = 0
    attempt_count = 0
    
    # Create fractions array with safety check
    fractions = np.arange(0, 1.0, step)
    if len(fractions) == 0:
        logger.warning(f"No fractions generated with step={step}, adjusting step size")
        fractions = np.linspace(0, 0.999, max(int(1/step), 10))
    
    # Set up iterator with or without progress bar
    iterator = tqdm(fractions, desc="Computing manifold") if show_progress else fractions
    
    for frac in iterator:
        attempt_count += 1
        try:
            # Get the initial condition on the manifold
            x0W = _compute_manifold_section(x0, T, frac, stbl, direction, mu, forward=forward)
            
            # Define integration time
            tf = integration_fraction * (2 * np.pi)
            
            # Ensure x0W is flattened and properly typed
            x0W_flat = x0W.flatten().astype(np.float64)
            
            # Propagate the trajectory
            logger.debug(f"Propagating manifold trajectory at fraction {frac:.3f} for time {tf:.3f}")
            sol = propagate_crtbp(x0W_flat, 0.0, tf, mu, forward=forward, steps=steps, **solver_kwargs)
            
            if not sol.success:
                logger.warning(f"Integration failed at fraction {frac:.3f} with message: {sol.message}")
                continue
                
            # Extract state and time vectors
            xW = sol.y.T
            tW = sol.t
            
            # Store trajectory
            xW_list.append(xW)
            tW_list.append(tW)

            # Compute the Poincaré section
            Xy0, Ty0 = _surface_of_section(xW, tW, mu, 2)
            
            if len(Xy0) > 0:
                # If intersection found, extract coordinates
                Xy0 = Xy0.flatten()
                ysos.append(Xy0[1])
                ydsos.append(Xy0[4])
                success_count += 1
                logger.debug(f"Fraction {frac:.3f}: Found Poincaré section point at y={Xy0[1]:.6f}, vy={Xy0[4]:.6f}")
            else:
                logger.warning(f"No section points found for fraction {frac:.3f}")
                
        except Exception as e:
            logger.error(f"Error processing fraction {frac:.3f}: {str(e)}", exc_info=True)
            continue
    
    logger.info(f"Manifold computation completed. Success rate: {success_count}/{attempt_count} points ({success_count/max(1, attempt_count)*100:.1f}%)")
    
    # Return results as a dataclass
    return ManifoldResult(
        ysos=ysos, 
        ydsos=ydsos, 
        xW_list=xW_list, 
        tW_list=tW_list,
        success_count=success_count,
        attempt_count=attempt_count
    )


def _compute_manifold_section(x0: np.ndarray, T: float, frac: float, stbl: int, 
                              direction: int, mu: float, NN: int = 1, 
                              forward: int = 1) -> np.ndarray:
    """
    Computes the initial condition on a stable or unstable manifold of a periodic orbit.
    
    This function computes the state transition matrix (STM) of the periodic orbit,
    performs eigendecomposition to find stable/unstable directions, and constructs
    a perturbed initial condition on the chosen manifold.
    
    Parameters
    ----------
    x0 : array_like
        Initial reference point on the periodic orbit (6D state).
    T : float
        Period of the orbit in non-dimensional CR3BP time.
    frac : float
        Fraction (0 to 1) along the orbit at which to compute the manifold.
    stbl : {1, -1}
        Manifold type selector:
        * 1 for stable manifold
        * -1 for unstable manifold
    direction : {1, -1}
        Branch selector:
        * 1 for the "positive" branch
        * -1 for the "negative" branch
    mu : float
        Mass ratio in the CR3BP.
    NN : int, optional
        If stable/unstable subspace is >1D, selects the NN-th real eigendirection.
        Default is 1.
    forward : {1, -1}, optional
        Direction of time integration for compute_stm. Default is 1 (forward in time).

    Returns
    -------
    x0W : ndarray
        A 6D state vector on the chosen (un)stable manifold of the periodic orbit.
    
    Raises
    ------
    ValueError
        If no real eigendirections are found or parameters are invalid.
    RuntimeError
        If numerical issues are encountered during computation.
    
    Notes
    -----
    The function follows these steps:
    1. Integrates the orbit with variational equations to get the state transition matrix
    2. Finds the time point corresponding to the specified fraction of the orbit
    3. Computes eigenvalues and eigenvectors of the monodromy matrix
    4. Based on eigenvalue magnitude, classifies them into stable, unstable, or center
    5. Applies the STM to project the eigenvector to the specified point on the orbit
    6. Perturbs the state along this direction with a small displacement (1e-6)
    """
    logger.debug(f"Computing manifold section at fraction {frac:.3f} of orbit")
    
    # Input validation
    if not (0 <= frac < 1):
        raise ValueError(f"Fraction must be in range [0, 1), got {frac}")
    
    if NN < 1:
        raise ValueError(f"NN must be a positive integer, got {NN}")
        
    # 1) Integrate to get monodromy and the full STM states
    try:
        xx, tt, phi_T, PHI = _compute_stm(x0, mu, T, forward=forward)
        logger.debug(f"STM computed with {len(tt)} time points")
    except Exception as e:
        logger.error(f"Failed to compute STM: {str(e)}")
        raise RuntimeError(f"STM computation failed: {str(e)}") from e

    # 2) Decompose the final monodromy to get stable/unstable eigenvectors
    try:
        sn, un, cn, y1Ws, y1Wu, y1Wc = _eig_decomp(phi_T, discrete=1)
        logger.debug(f"Eigendecomposition complete. Found {len(sn)} stable, {len(un)} unstable, {len(cn)} center eigenvalues")
    except Exception as e:
        logger.error(f"Eigendecomposition failed: {str(e)}")
        raise RuntimeError(f"Eigendecomposition failed: {str(e)}") from e

    # 3) Collect real eigen-directions for stable set
    snreal_vals = []
    snreal_vecs = []
    for k in range(len(sn)):
        if np.isreal(sn[k]):
            snreal_vals.append(sn[k])
            snreal_vecs.append(y1Ws[:, k])

    # 4) Collect real eigen-directions for unstable set
    unreal_vals = []
    unreal_vecs = []
    for k in range(len(un)):
        if np.isreal(un[k]):
            unreal_vals.append(un[k])
            unreal_vecs.append(y1Wu[:, k])

    # Log real eigenvalues found
    logger.debug(f"Found {len(snreal_vals)} real stable eigenvalues: {snreal_vals}")
    logger.debug(f"Found {len(unreal_vals)} real unstable eigenvalues: {unreal_vals}")

    # Convert lists to arrays
    snreal_vals = np.array(snreal_vals, dtype=np.complex128)
    unreal_vals = np.array(unreal_vals, dtype=np.complex128)
    snreal_vecs = (np.column_stack(snreal_vecs) 
                   if len(snreal_vecs) else np.zeros((6, 0), dtype=np.complex128))
    unreal_vecs = (np.column_stack(unreal_vecs) 
                   if len(unreal_vecs) else np.zeros((6, 0), dtype=np.complex128))

    # 5) Select the NN-th real eigendirection (MATLAB is 1-based, Python is 0-based)
    col_idx = NN - 1
    
    # Check if requested eigenvector exists
    if stbl == 1 and (snreal_vecs.shape[1] <= col_idx or col_idx < 0):
        raise ValueError(f"Requested stable eigenvector {NN} not available. Only {snreal_vecs.shape[1]} real stable eigenvectors found.")
    
    if stbl == -1 and (unreal_vecs.shape[1] <= col_idx or col_idx < 0):
        raise ValueError(f"Requested unstable eigenvector {NN} not available. Only {unreal_vecs.shape[1]} real unstable eigenvectors found.")
    
    # Select appropriate eigenvector
    WS = snreal_vecs[:, col_idx] if stbl == 1 else None
    WU = unreal_vecs[:, col_idx] if stbl == -1 else None

    # 6) Find the row index for t ~ frac*T
    try:
        mfrac = _totime(tt, frac * T)  # integer index
        logger.debug(f"Selected time point {mfrac} at t={tt[mfrac]:.6f} for fraction {frac:.3f}")
    except Exception as e:
        logger.error(f"Failed to find time point for fraction {frac}: {str(e)}")
        raise RuntimeError(f"Time mapping failed: {str(e)}") from e

    # 7) Reshape PHI to get the 6x6 STM at that time.
    try:
        phi_frac_flat = PHI[mfrac, :36]  # first 36 columns
        phi_frac = phi_frac_flat.reshape((6, 6))
    except Exception as e:
        logger.error(f"Failed to reshape STM: {str(e)}")
        raise RuntimeError(f"STM reshaping failed: {str(e)}") from e

    # 8) Decide stable vs. unstable direction and compute manifold direction
    try:
        if stbl == +1:   # stable manifold
            MAN = direction * (phi_frac @ WS)
            logger.debug(f"Using stable manifold direction with eigenvalue {snreal_vals[col_idx]:.6f}")
        else:            # unstable manifold
            MAN = direction * (phi_frac @ WU)
            logger.debug(f"Using unstable manifold direction with eigenvalue {unreal_vals[col_idx]:.6f}")
    except Exception as e:
        logger.error(f"Failed to compute manifold direction: {str(e)}")
        raise RuntimeError(f"Manifold direction computation failed: {str(e)}") from e

    # 9) Scale the displacement (hard-coded 1e-6 factor)
    disp_magnitude = np.linalg.norm(MAN[0:3])
    if disp_magnitude < 1e-14:
        logger.warning(f"Very small displacement magnitude: {disp_magnitude:.2e}, setting to 1.0")
        disp_magnitude = 1.0
    d = 1e-6 / disp_magnitude
    logger.debug(f"Displacement scaling factor: {d:.6e}")

    # 10) Reference orbit state at t=tt[mfrac]
    fracH = xx[mfrac, :].copy()  # shape (6,)

    # 11) Construct the final manifold state
    x0W = fracH + d * MAN.real  # ensure real if there's a tiny imaginary part
    x0W = x0W.flatten()
    
    # 12) Zero out tiny numerical noise
    if abs(x0W[2]) < 1.0e-15:
        x0W[2] = 0.0
    if abs(x0W[5]) < 1.0e-15:
        x0W[5] = 0.0

    logger.debug(f"Computed manifold point: {x0W}")
    return x0W