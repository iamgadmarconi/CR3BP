import numba
import numpy as np


@numba.njit(fastmath=True, cache=True)
def crtbp_accel(state, mu):
    """
    State = [x, y, z, vx, vy, vz]
    Returns the time derivative of the state vector for the CRTBP.
    """
    x, y, z, vx, vy, vz = state

    # Distances to each primary
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)      # from m1 at (-mu, 0, 0)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2) # from m2 at (1-mu, 0, 0)

    # Accelerations
    ax = 2*vy + x - (1 - mu)*(x + mu) / r1**3 - mu*(x - 1 + mu) / r2**3
    ay = -2*vx + y - (1 - mu)*y / r1**3          - mu*y / r2**3
    az = -(1 - mu)*z / r1**3 - mu*z / r2**3

    return np.array([vx, vy, vz, ax, ay, az], dtype=np.float64)

@numba.njit(fastmath=True, cache=True)
def jacobian_crtbp(x, y, z, mu):
    """
    Returns the 6x6 Jacobian matrix F for the 3D CRTBP in the rotating frame,
    mirroring the MATLAB var3D.m approach (with mu2 = 1 - mu).
    
    The matrix F is structured as:
         [ 0    0    0    1     0    0 ]
         [ 0    0    0    0     1    0 ]
         [ 0    0    0    0     0    1 ]
         [ omgxx omgxy omgxz  0     2    0 ]
         [ omgxy omgyy omgyz -2     0    0 ]
         [ omgxz omgyz omgzz  0     0    0 ]

    Indices: x=0, y=1, z=2, vx=3, vy=4, vz=5

    This matches the partial derivatives from var3D.m exactly.
    """

    # As in var3D.m:
    #   mu2 = 1 - mu (big mass fraction)
    mu2 = 1.0 - mu

    # Distances squared to the two primaries
    # r^2 = (x+mu)^2 + y^2 + z^2       (distance^2 to M1, which is at (-mu, 0, 0))
    # R^2 = (x - mu2)^2 + y^2 + z^2    (distance^2 to M2, which is at (1-mu, 0, 0))
    r2 = (x + mu)**2 + y**2 + z**2
    R2 = (x - mu2)**2 + y**2 + z**2
    r3 = r2**1.5
    r5 = r2**2.5
    R3 = R2**1.5
    R5 = R2**2.5

    # From var3D.m, the partial derivatives "omgxx," "omgyy," ...
    omgxx = 1.0 \
        + mu2/r5 * 3.0*(x + mu)**2 \
        + mu  /R5 * 3.0*(x - mu2)**2 \
        - (mu2/r3 + mu/R3)

    omgyy = 1.0 \
        + mu2/r5 * 3.0*(y**2) \
        + mu  /R5 * 3.0*(y**2) \
        - (mu2/r3 + mu/R3)

    omgzz = 0.0 \
        + mu2/r5 * 3.0*(z**2) \
        + mu  /R5 * 3.0*(z**2) \
        - (mu2/r3 + mu/R3)

    omgxy = 3.0*y * ( mu2*(x + mu)/r5 + mu*(x - mu2)/R5 )
    omgxz = 3.0*z * ( mu2*(x + mu)/r5 + mu*(x - mu2)/R5 )
    omgyz = 3.0*y*z*( mu2/r5 + mu/R5 )

    # Build the 6x6 matrix F
    F = np.zeros((6, 6), dtype=np.float64)

    # Identity block for velocity wrt position
    F[0, 3] = 1.0  # dx/dvx
    F[1, 4] = 1.0  # dy/dvy
    F[2, 5] = 1.0  # dz/dvz

    # The second derivatives block
    F[3, 0] = omgxx
    F[3, 1] = omgxy
    F[3, 2] = omgxz

    F[4, 0] = omgxy
    F[4, 1] = omgyy
    F[4, 2] = omgyz

    F[5, 0] = omgxz
    F[5, 1] = omgyz
    F[5, 2] = omgzz

    # Coriolis terms
    F[3, 4] = 2.0
    F[4, 3] = -2.0

    return F

@numba.njit(fastmath=True, cache=True)
def variational_equations(t, PHI_vec, mu, forward=1):
    """
    3D variational equations for the CR3BP, matching MATLAB's var3D.m layout.
    
    PHI_vec is a 42-element vector:
      - PHI_vec[:36]   = the flattened 6x6 STM (Phi)
      - PHI_vec[36:42] = the state vector [x, y, z, vx, vy, vz]
    
    We compute d/dt(PHI_vec).
    The resulting derivative is also 42 elements:
      - first 36 = dPhi/dt (flattened)
      - last 6 = [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt] * forward

    This function calls 'jacobian_crtbp_matlab(...)' to build the 6x6 matrix F
    and then does the same matrix multiplication as in var3D.m:

        phidot = F * Phi
    """
    # 1) Unpack the STM (first 36) and the state (last 6)
    phi_flat = PHI_vec[:36]
    x_vec    = PHI_vec[36:]  # [x, y, z, vx, vy, vz]

    # Reshape the STM to 6x6
    Phi = phi_flat.reshape((6, 6))

    # Unpack the state
    x, y, z, vx, vy, vz = x_vec

    # 2) Build the 6x6 matrix F from the partial derivatives
    F = jacobian_crtbp(x, y, z, mu)

    # 3) dPhi/dt = F * Phi  (manually done to keep numba happy)
    phidot = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            s = 0.0 
            for k in range(6):
                s += F[i, k] * Phi[k, j]
            phidot[i, j] = s

    # 4) State derivatives, same formula as var3D.m
    #    xdot(4) = x(1) - mu2*( (x+mu)/r3 ) - mu*( (x-mu2)/R3 ) + 2*vy, etc.
    mu2 = 1.0 - mu
    r2 = (x + mu)**2 + y**2 + z**2
    R2 = (x - mu2)**2 + y**2 + z**2
    r3 = r2**1.5
    R3 = R2**1.5

    ax = ( x 
           - mu2*( (x+mu)/r3 ) 
           -  mu*( (x - mu2)/R3 ) 
           + 2.0*vy )
    ay = ( y
           - mu2*( y / r3 )
           -  mu*( y / R3 )
           - 2.0*vx )
    az = ( - mu2*( z / r3 ) 
           - mu  *( z / R3 ) )

    # 5) Build derivative of the 42-vector
    dPHI_vec = np.zeros_like(PHI_vec)

    # First 36 = flattened phidot
    dPHI_vec[:36] = phidot.ravel()

    # Last 6 = [vx, vy, vz, ax, ay, az], each multiplied by 'forward'
    dPHI_vec[36] = forward * vx
    dPHI_vec[37] = forward * vy
    dPHI_vec[38] = forward * vz
    dPHI_vec[39] = forward * ax
    dPHI_vec[40] = forward * ay
    dPHI_vec[41] = forward * az

    return dPHI_vec