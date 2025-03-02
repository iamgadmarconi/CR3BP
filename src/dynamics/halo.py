import numpy as np

def gammaL(mu, Lpt):
    """
    Calculate ratio of libration point distance from closest primary to distance
    between two primaries (example gammaL1 = (E-L1)/AU)
    
    Parameters:
    -----------
    mu : float
        Mass parameter of system [mu = m2/(m1+m2)]
    Lpt : int
        Lagrange point number (1, 2, or 3)
        
    Returns:
    --------
    gamma : float
        Ratio of distances
    """
    mu2 = 1 - mu
    
    poly1 = [1, -1*(3-mu), (3-2*mu), -mu, 2*mu, -mu]
    poly2 = [1, (3-mu), (3-2*mu), -mu, -2*mu, -mu]
    poly3 = [1, (2+mu), (1+2*mu), -mu2, -2*mu2, -mu2]
    
    rt1 = np.roots(poly1)
    rt2 = np.roots(poly2)
    rt3 = np.roots(poly3)
    
    GAMMAS = [0, 0, 0]  # Initialize with zeros (not necessary but matches MATLAB structure)
    
    for k in range(5):
        if np.isreal(rt1[k]):
            GAMMAS[0] = np.real(rt1[k])  # Convert to real since isreal() may return True for complex with zero imag
        if np.isreal(rt2[k]):
            GAMMAS[1] = np.real(rt2[k])
        if np.isreal(rt3[k]):
            GAMMAS[2] = np.real(rt3[k])
    
    return GAMMAS[Lpt-1]  # Adjust for 0-indexing in Python

def halogesANL(mu, Lpt, Azlp, n):
    """
    Gives initial state (position,velocity) for a 3D periodic halo orbit 
    centered on the specified collinear Lagrange point.
    [Uses Richardson's 3rd order model for analytically constructing a 3D 
     periodic halo orbit about the points L1, L2, or L3]
    
    Parameters:
    -----------
    mu : float
        Mass parameter of system [mu = m2/(m1+m2)]
    Lpt : int
        Lagrange point number (1, 2, or 3)
    Azlp : float
        Out-of-plane (or z-amplitude) of the desired halo
        [in Lpt-primary distances]
    n : int
        +1 is northern halo (z>0, Class I),
        -1 is southern halo (z<0, Class II)
        
    Returns:
    --------
    x0 : ndarray
        Initial conditions (position and velocity vectors) for the
        desired halo orbit (in 3D CR3BP nondim. units)
    """
    Az = Azlp
    
    gamma = gammaL(mu, Lpt)
    if Lpt == 1:
        won = +1
        primary = 1-mu
    elif Lpt == 2:
        won = -1
        primary = 1-mu
    elif Lpt == 3:
        won = +1
        primary = -mu
    
    c = [0, 0, 0, 0, 0]  # Create array with indices 0-4 (though we'll only use 2-4)
    
    if Lpt == 3:
        for N in range(2, 5):  # N = 2,3,4
            c[N] = (1/gamma**3) * (1-mu + (-primary*gamma**(N+1))/((1+gamma)**(N+1)))
    else:
        for N in range(2, 5):  # N = 2,3,4
            c[N] = (1/gamma**3) * ((won**N)*mu + ((-1)**N)*((primary)*gamma**(N+1))/((1+(-won)*gamma)**(N+1)))
    
    polylambda = [1, 0, (c[2]-2), 0, -(c[2]-1)*(1+2*c[2])]
    lambda_vals = np.roots(polylambda)  # lambda = frequency of orbit
    
    if Lpt == 3:
        lambda_val = abs(lambda_vals[2])
    else:
        lambda_val = abs(lambda_vals[0])
    
    k = 2*lambda_val/(lambda_val**2 + 1 - c[2])
    del_val = lambda_val**2 - c[2]
    
    d1 = ((3*lambda_val**2)/k)*(k*(6*lambda_val**2 - 1) - 2*lambda_val)
    d2 = ((8*lambda_val**2)/k)*(k*(11*lambda_val**2 - 1) - 2*lambda_val)
    
    a21 = (3*c[3]*(k**2 - 2))/(4*(1 + 2*c[2]))
    a22 = 3*c[3]/(4*(1 + 2*c[2]))
    a23 = -(3*c[3]*lambda_val/(4*k*d1))*( 3*(k**3)*lambda_val - 6*k*(k-lambda_val) + 4)
    a24 = -(3*c[3]*lambda_val/(4*k*d1))*( 2 + 3*k*lambda_val )
    
    b21 = -(3*c[3]*lambda_val/(2*d1))*(3*k*lambda_val - 4)
    b22 = 3*c[3]*lambda_val/d1
    d21 = -c[3]/(2*lambda_val**2)
    
    a31 = -(9*lambda_val/(4*d2))*(4*c[3]*(k*a23 - b21) + k*c[4]*(4 + k**2)) + ((9*lambda_val**2 + 1 - c[2])/(2*d2))*(3*c[3]*(2*a23 - k*b21) + c[4]*(2 + 3*k**2))
    a32 = -(1/d2)*( (9*lambda_val/4)*(4*c[3]*(k*a24 - b22) + k*c[4]) + 1.5*(9*lambda_val**2 + 1 - c[2])*( c[3]*(k*b22 + d21 - 2*a24) - c[4]) )
    
    b31 = (.375/d2)*( 8*lambda_val*(3*c[3]*(k*b21 - 2*a23) - c[4]*(2 + 3*k**2)) + (9*lambda_val**2 + 1 + 2*c[2])*(4*c[3]*(k*a23 - b21) + k*c[4]*(4 + k**2)) )
    b32 = (1/d2)*( 9*lambda_val*(c[3]*(k*b22 + d21 - 2*a24) - c[4]) + .375*(9*lambda_val**2 + 1 + 2*c[2])*(4*c[3]*(k*a24 - b22) + k*c[4]) )
    
    d31 = (3/(64*lambda_val**2))*(4*c[3]*a24 + c[4])
    d32 = (3/(64*lambda_val**2))*(4*c[3]*(a23 - d21) + c[4]*(4 + k**2))
    
    s1 = (1/(2*lambda_val*(lambda_val*(1+k**2) - 2*k)))*( 1.5*c[3]*(2*a21*(k**2 - 2)-a23*(k**2 + 2) - 2*k*b21) - .375*c[4]*(3*k**4 - 8*k**2 + 8) )
    s2 = (1/(2*lambda_val*(lambda_val*(1+k**2) - 2*k)))*( 1.5*c[3]*(2*a22*(k**2 - 2)+a24*(k**2 + 2) + 2*k*b22 + 5*d21) + .375*c[4]*(12 - k**2) )
    
    a1 = -1.5*c[3]*(2*a21 + a23 + 5*d21) - .375*c[4]*(12-k**2)
    a2 = 1.5*c[3]*(a24-2*a22) + 1.125*c[4]
    
    l1 = a1 + 2*(lambda_val**2)*s1
    l2 = a2 + 2*(lambda_val**2)*s2
    
    tau1 = 0
    deltan = -n
    
    Ax = np.sqrt((-del_val - l2*Az**2)/l1)
    
    x = a21*Ax**2 + a22*Az**2 - Ax*np.cos(tau1) + (a23*Ax**2 - a24*Az**2)*np.cos(2*tau1) + (a31*Ax**3 - a32*Ax*Az**2)*np.cos(3*tau1)
    y = k*Ax*np.sin(tau1) + (b21*Ax**2 - b22*Az**2)*np.sin(2*tau1) + (b31*Ax**3 - b32*Ax*Az**2)*np.sin(3*tau1)
    z = deltan*Az*np.cos(tau1) + deltan*d21*Ax*Az*(np.cos(2*tau1) - 3) + deltan*(d32*Az*Ax**2 - d31*Az**3)*np.cos(3*tau1)
    
    xdot = lambda_val*Ax*np.sin(tau1) - 2*lambda_val*(a23*Ax**2 - a24*Az**2)*np.sin(2*tau1) - 3*lambda_val*(a31*Ax**3 - a32*Ax*Az**2)*np.sin(3*tau1)
    ydot = lambda_val*(k*Ax*np.cos(tau1) + 2*(b21*Ax**2 - b22*Az**2)*np.cos(2*tau1) + 3*(b31*Ax**3 - b32*Ax*Az**2)*np.cos(3*tau1))
    zdot = -lambda_val*deltan*Az*np.sin(tau1) - 2*lambda_val*deltan*d21*Ax*Az*np.sin(2*tau1) - 3*lambda_val*deltan*(d32*Az*Ax**2 - d31*Az**3)*np.sin(3*tau1)
    
    r0 = gamma * np.array([(primary + gamma*(-won + x))/gamma, -y, z])
    v0 = gamma * np.array([xdot, ydot, zdot])
    x0 = np.concatenate((r0, v0))
    
    return x0
