"""
Utility functions for numerical operations on manifolds in the CR3BP.

This module provides a collection of helper functions that support numerical
computations when analyzing manifolds in the Circular Restricted Three-Body Problem.
It includes functions for handling numerical precision issues, finding indices in
time arrays, and interpolating trajectory data.

These utilities are designed to ensure numerical stability and accuracy in
dynamical systems calculations, particularly when dealing with complex eigenvalues
and eigenvectors, or when precise interpolation of trajectory data is required.
"""

import numpy as np
from scipy.interpolate import CubicSpline


def _remove_infinitesimals_in_place(vec, tol=1e-14):
    """
    Replace very small real and imaginary parts with exact zeros in-place.
    
    This function modifies the input vector by setting real and imaginary parts
    that are smaller than the tolerance to exactly zero. This helps prevent
    numerical noise from affecting calculations.
    
    Parameters
    ----------
    vec : array_like
        Complex vector to be modified in-place
    tol : float, optional
        Tolerance level below which values are set to zero.
        Default is 1e-14.
    
    Returns
    -------
    None
        The input vector is modified in-place.
    
    Notes
    -----
    This function is particularly useful for cleaning up eigenvectors that might
    have tiny numerical artifacts in their components.
    """
    for i in range(len(vec)):
        re = vec[i].real
        im = vec[i].imag
        if abs(re) < tol:
            re = 0.0
        if abs(im) < tol:
            im = 0.0
        vec[i] = re + 1j*im

def _remove_infinitesimals_array(vec, tol=1e-12):
    """
    Create a copy of a vector with very small values replaced by exact zeros.
    
    This function creates a copy of the input vector and then calls
    _remove_infinitesimals_in_place on the copy. This preserves the original
    vector while returning a "cleaned" version.
    
    Parameters
    ----------
    vec : array_like
        Complex vector to be cleaned
    tol : float, optional
        Tolerance level below which values are set to zero.
        Default is 1e-12.
    
    Returns
    -------
    ndarray
        A copy of the input vector with small values replaced by zeros
    
    See Also
    --------
    _remove_infinitesimals_in_place : The in-place version of this function
    """
    vcopy = vec.copy()
    _remove_infinitesimals_in_place(vcopy, tol)
    return vcopy

def _zero_small_imag_part(eig_val, tol=1e-12):
    """
    Set the imaginary part of a complex number to zero if it's very small.
    
    This function is useful for cleaning up eigenvalues that should be real
    but might have tiny imaginary components due to numerical precision issues.
    
    Parameters
    ----------
    eig_val : complex
        Complex value to be checked and potentially modified
    tol : float, optional
        Tolerance level below which the imaginary part is set to zero.
        Default is 1e-12.
    
    Returns
    -------
    complex
        The input value with its imaginary part set to zero if it was smaller
        than the tolerance
    """
    if abs(eig_val.imag) < tol:
        return complex(eig_val.real, 0.0)
    return eig_val

def _totime(t, tf):
    """
    Find indices in a time array that are closest to specified target times.
    
    This function searches through a time array and finds the indices where
    the time values are closest to the specified target times.
    
    Parameters
    ----------
    t : array_like
        Array of time values to search through
    tf : float or array_like
        Target time value(s) to find in the array
    
    Returns
    -------
    ndarray
        Array of indices where the values in 't' are closest to the
        corresponding values in 'tf'
    
    Notes
    -----
    This function is particularly useful when trying to find the point in a
    trajectory closest to a specific time, such as when identifying positions
    at specific fractions of a periodic orbit.
    
    The function works with absolute time values, so the sign of the input times
    does not affect the results.
    """
    # Convert t to its absolute values.
    t = np.abs(t)
    # Ensure tf is an array (handles scalar input as well).
    tf = np.atleast_1d(tf)
    
    # Preallocate an array to hold the indices.
    I = np.empty(tf.shape, dtype=int)
    
    # For each target value in tf, find the index in t closest to it.
    for k, target in enumerate(tf):
        diff = np.abs(target - t)
        I[k] = np.argmin(diff)
    
    return I

def _interpolate(x, t, dt=None):
    """
    Re-sample a trajectory using cubic spline interpolation.
    
    This function takes trajectory data with potentially unevenly spaced time points
    and produces a new trajectory with evenly spaced time points using cubic spline
    interpolation. It handles both scalar and vector-valued trajectories.
    
    Parameters
    ----------
    x : ndarray
        Data to interpolate, with shape (m, n) where each row is a state vector
        at a particular time, and each column is a variable tracked over time.
        For a scalar trajectory, x can be a 1D array of length m.
    t : ndarray
        Original time points corresponding to the rows of x, with shape (m,)
    dt : float or int, optional
        Either:
        * If dt <= 10: the time step between interpolated points
        * If dt > 10: the number of points desired in the output
        If not provided, defaults to 0.05 * 2π.
    
    Returns
    -------
    X : ndarray
        Interpolated data with evenly spaced time steps. Shape is (N, n) for
        vector-valued input or (N,) for scalar input, where N is the number
        of interpolated points.
    T : ndarray
        New time vector corresponding to the rows of X, with shape (N,)
    
    Notes
    -----
    The function handles time vectors that span both negative and positive values,
    preserving the "arrow of time" (direction of time flow) in the output.
    
    This is particularly useful for:
    - Creating smoother visualizations of trajectories
    - Ensuring consistent time steps for numerical analysis
    - Finding precise crossing times in Poincaré section analysis
    """
    t = np.asarray(t)
    x = np.asarray(x)
    
    # Default dt if not provided
    if dt is None:
        dt = 0.05 * 2 * np.pi

    # If dt > 10, then treat dt as number of points (N) and recalc dt
    if dt > 10:
        N = int(dt)
        dt = (np.max(t) - np.min(t)) / (N - 1)
    
    # Adjust time vector if it spans negative and positive values
    NEG = 1 if (np.min(t) < 0 and np.max(t) > 0) else 0
    tt = np.abs(t - NEG * np.min(t))
    
    # Create new evenly spaced time vector for the interpolation domain
    TT = np.arange(tt[0], tt[-1] + dt/10, dt)
    # Recover the correct "arrow of time"
    T = np.sign(t[-1]) * TT + NEG * np.min(t)
    
    # Interpolate each column using cubic spline interpolation
    if x.ndim == 1:
        # For a single-dimensional x, treat as a single column
        cs = CubicSpline(tt, x)
        X = cs(TT)
    else:
        m, n = x.shape
        X = np.zeros((len(TT), n))
        for i in range(n):
            cs = CubicSpline(tt, x[:, i])
            X[:, i] = cs(TT)
    
    return X, T