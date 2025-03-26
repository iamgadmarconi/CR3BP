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

def _interpolate(x, t=None, dt=None):
    """
    Function with dual behavior:
    1. When called with 3 arguments like _interpolate(x1, x2, s), performs simple linear interpolation
       between two points x1 and x2 with parameter s.
    2. When called with trajectory data _interpolate(x, t, dt), resamples a trajectory using cubic splines.
    
    Parameters
    ----------
    x : ndarray
        Either:
        * First argument: Data to interpolate for trajectory resampling
        * First point x1 for simple interpolation
    t : ndarray or float, optional
        Either:
        * Time points for trajectory resampling
        * Second point x2 for simple interpolation
    dt : float or int, optional
        Either:
        * Time step or number of points for trajectory resampling
        * Interpolation parameter s for simple interpolation
    
    Returns
    -------
    X : ndarray or tuple
        Either:
        * Interpolated point between x1 and x2 (for simple interpolation)
        * Tuple (X, T) of resampled trajectory and time vector (for trajectory resampling)
    
    Notes
    -----
    This function determines which behavior to use based on the number and types
    of arguments provided. For backward compatibility, it supports both the original
    trajectory resampling behavior and the simple point interpolation used in
    surface_of_section calculations.
    """
    # Special case: When called with 3 arguments from surface_of_section
    # Using pattern: _interpolate(X1, X2, s)
    # where s is a scalar in [0, 1]
    if dt is not None and np.isscalar(dt) and (0 <= dt <= 1):
        # This is simple linear interpolation
        # x = x1, t = x2, dt = s
        s = dt
        x1 = x
        x2 = t
        
        # Ensure s is in [0, 1]
        s = max(0, min(1, s))
        
        # Simple linear interpolation
        return x1 + s * (x2 - x1)
        
    # Original trajectory resampling case
    t = np.asarray(t) if t is not None else None
    x = np.asarray(x)
    
    # Default dt if not provided
    if dt is None:
        dt = 0.05 * 2 * np.pi
    
    # Handle special cases for t
    if t is None or len(t) < 2:
        return x  # Can't interpolate

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