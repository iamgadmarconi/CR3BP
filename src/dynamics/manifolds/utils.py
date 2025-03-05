import numpy as np
from scipy.interpolate import CubicSpline


def _remove_infinitesimals_in_place(vec, tol=1e-14):
    for i in range(len(vec)):
        re = vec[i].real
        im = vec[i].imag
        if abs(re) < tol:
            re = 0.0
        if abs(im) < tol:
            im = 0.0
        vec[i] = re + 1j*im

def _remove_infinitesimals_array(vec, tol=1e-12):
    vcopy = vec.copy()
    _remove_infinitesimals_in_place(vcopy, tol)
    return vcopy

def _zero_small_imag_part(eig_val, tol=1e-12):
    if abs(eig_val.imag) < tol:
        return complex(eig_val.real, 0.0)
    return eig_val

def _totime(t, tf):
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
    Re-samples a trajectory x with time stamps t using cubic spline interpolation.
    
    Parameters:
      x  : ndarray of shape (m, n) -- data to interpolate (each column is a variable)
      t  : 1D array of length m   -- original time points
      dt : Either the time step between interpolated points or,
           if dt > 10, the number of points desired.
           If not provided, defaults to 0.05 * 2*pi.
    
    Returns:
      X  : ndarray -- interpolated data with evenly spaced time steps
      T  : 1D array -- new time vector corresponding to X
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