import numba
import numpy as np
from scipy.integrate import solve_ivp

from .propagator import crtbp_accel


import numpy as np

def halo_diff_correct(x0_guess, mu, tol=1e-12, max_iter=25):
    """
    3D "Halo" differential corrector that:
      - Fixes x0 (the initial x-position).
      - Varies z0 and vy0.
      - Enforces next crossing y=0 to have vx=0, vz=0 (i.e. symmetrical crossing).
    Returns (x0_corrected, half_period).
    
    x0_guess is [ x, 0, z, vx, vy, vz ].
    We'll update z and vy as needed.
    """
    x0 = np.copy(x0_guess)
    
    iteration = 0
    done = False
    while not done:
        iteration += 1
        if iteration > max_iter:
            raise RuntimeError("Halo corrector failed to converge within max_iter.")
        
        # 1) Integrate to the next y=0 crossing
        #    We'll guess a half-period near pi or 3.0... up to you:
        guess_t = np.pi
        t_cross, X_cross = find_y_crossing(x0, mu, guess_t, direction=+1, tol=tol)
        
        # Evaluate the crossing conditions:
        vx_cross = X_cross[3]  # dot{x}
        vy_cross = X_cross[4]
        vz_cross = X_cross[5]  # dot{z}
        
        # We want dx1=0 => vx_cross=0 and dz1=0 => vz_cross=0
        if np.sqrt(vx_cross**2 + vz_cross**2) < tol:
            # Done
            return x0, t_cross
        
        # 2) We do a 2-variable Newton step: unknowns are [z0, vy0].
        #    We have 2 constraints: vx_cross=0, vz_cross=0.
        #    We'll compute partial derivatives from the STM at t_cross.
        _, final_state, Phi_final = compute_stm(x0, 
                                                mu, 
                                                t_cross, 
                                                atol=tol, 
                                                rtol=tol)
        # final_state must match X_cross
        # Phi_final is the 6x6 STM from t=0 to t=t_cross.
        
        # The partial derivatives we care about:
        #   dvx/dz0 = Phi_final(3, 2)
        #   dvx/dvy0= Phi_final(3, 4)
        #   dvz/dz0 = Phi_final(5, 2)
        #   dvz/dvy0= Phi_final(5, 4)
        #
        # Indices in 0-based Python:
        #   row=3 => vx, row=5 => vz
        #   col=2 => z0, col=4 => vy0
        dvx_dz0  = Phi_final[3,2]
        dvx_dvy0 = Phi_final[3,4]
        dvz_dz0  = Phi_final[5,2]
        dvz_dvy0 = Phi_final[5,4]
        
        # So we have constraints F = [vx_cross, vz_cross], unknowns U = [z0, vy0].
        # We'll do a 2×2 system: dF/dU = [[dvx_dz0, dvx_dvy0],
        #                                [dvz_dz0, dvz_dvy0]]
        # solve  dU = - inv(dF/dU) * F
        # where F = [vx_cross, vz_cross].
        DF = np.array([
            [dvx_dz0,  dvx_dvy0],
            [dvz_dz0,  dvz_dvy0]
        ])
        Fvals = np.array([vx_cross, vz_cross])
        
        dU = -np.linalg.solve(DF, Fvals)
        
        dz0  = dU[0]
        dvy0 = dU[1]
        
        # Update x0: (we do not change x0[0], because we fix x!)
        x0[2] += dz0   # z0
        x0[4] += dvy0  # vy0

    # If we exit loop “normally,” just return
    return x0, t_cross


def lyapunov_diff_correct(x0_guess, mu, tol=1e-12, max_iter=50):
    """
    Differential corrector for a planar Lyapunov-like orbit.
    Keeps x0, y0, z0, vx0, vz0 fixed, adjusting only vy0
    so that the next crossing of y=0 has vx(t_cross) = 0 (or some condition).
    
    Returns (x0_corrected, half_period).
    """
    x0 = np.copy(x0_guess)
    attempt = 0
    
    while True:
        attempt += 1
        if attempt > max_iter:
            raise RuntimeError("Max attempts exceeded in differential corrector.")
        
        # 1) Find next y=0 crossing from some guess time. 
        #    We can guess half a period near pi, or adapt as you see fit
        guess_t = np.pi
        t_cross, X_cross = find_y_crossing(x0, mu, guess_t)
        
        # The crossing states
        x_cross = X_cross[0]
        y_cross = X_cross[1]
        z_cross = X_cross[2]
        vx_cross = X_cross[3]
        vy_cross = X_cross[4]
        vz_cross = X_cross[5]
        
        # Condition we want: e.g., vx_cross = 0 at y=0 crossing
        # If it's close to zero, we stop
        if abs(vx_cross) < tol:
            # Done
            half_period = t_cross
            return x0, half_period
        
        # Otherwise, we do a Newton step:
        # Evaluate the STM at t_cross to see how vx(t_cross) depends on vy0
        # We'll do the same integration but with the STM.
        PHI0 = np.zeros(42)
        PHI0[:6] = x0
        PHI0[6:] = np.eye(6).flatten()
        
        def ode_fun(t, y):
            return variational_equations(t, y, mu)
        
        # We'll integrate to the crossing time directly
        sol = solve_ivp(ode_fun, [0, t_cross], PHI0, rtol=1e-12, atol=1e-12, dense_output=True)
        
        # Evaluate the final 42-vector at t_cross
        PHI_vec_final = sol.sol(t_cross)
        # The final state:
        state_final = PHI_vec_final[:6]
        # Flattened STM
        phi_final = PHI_vec_final[6:].reshape((6,6))
        
        # We want partial of vx_cross wrt initial vy. That is phi_final(3,4) in 0-based indexing:
        # ( Row=3 for vx, Col=4 for vy ), if all other initial states are held fixed.
        dvx_dvy0 = phi_final[3,4]
        
        # Basic linear correction:  dvy = - vx_cross / dvx_dvy0
        dvy = - vx_cross / dvx_dvy0
        
        # Update x0
        x0[4] += dvy  # adjust the initial vy


def find_y_crossing(x0, mu, guess_t, direction=1, tol=1e-12, max_steps=1000):
    """
    Given an initial condition x0 (6D) and a guess for the crossing time guess_t,
    integrate and find the time t_cross when y(t) = 0 in the desired direction 
    (either from + to - or - to +).
    Returns (t_cross, X_cross).
    """
    def event_y_eq_zero(t, y):
        # Ensure we don’t trigger at t=0
        if t == 0:
            return 1.0  # anything nonzero so that it won't register as a root
        return y[1]

    event_y_eq_zero.direction = direction  # +1 or -1
    event_y_eq_zero.terminal = True
    
    sol = solve_ivp(lambda t, y: crtbp_accel(y, mu),
                    [0, guess_t], x0,
                    events=event_y_eq_zero, rtol=1e-12, atol=1e-12, dense_output=True)
    if len(sol.t_events[0]) == 0:
        # fallback: no crossing found in [0, guess_t], so try extending
        sol = solve_ivp(lambda t, y: crtbp_accel(y, mu),
                        [0, 10*guess_t], x0,
                        events=event_y_eq_zero, rtol=1e-12, atol=1e-12, dense_output=True)
        if len(sol.t_events[0]) == 0:
            raise RuntimeError("No y=0 crossing found. Try a bigger guess or different direction.")
    
    t_cross = sol.t_events[0][0]
    X_cross = sol.sol(t_cross)
    return t_cross, X_cross


def jacobian_crtbp(x, y, z, mu):
    """
    Returns the 6x6 Jacobian of the CR3BP vector field
    f(X) = [vx, vy, vz, ax, ay, az]
    with respect to X = [x, y, z, vx, vy, vz].
    
    This includes the partials of (ax, ay, az) wrt (x,y,z)
    PLUS the standard rotating-frame terms that couple velocity.
    """
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    
    # To avoid repeated calls:
    r1_3 = r1**3
    r1_5 = r1**5
    r2_3 = r2**3
    r2_5 = r2**5
    
    # Derivatives of the CR3BP acceleration in rotating frame:
    # ax = 2*vy + x - (1 - mu)*(x + mu)/r1^3 - mu*(x - 1 + mu)/r2^3
    # We want partial derivatives wrt x,y,z of ax, ay, az.
    
    # Common factors:
    mu1 = 1.0 - mu
    mu2 = mu
    
    # partial wrt x
    dUxx = (1
            - mu1*(1.0/r1_3 - 3.0*(x+mu)**2 / r1_5)
            - mu2*(1.0/r2_3 - 3.0*(x - 1 + mu)**2 / r2_5))
    dUxy = (0
            + 3.0*mu1*(x+mu)*y / r1_5
            + 3.0*mu2*(x-1+mu)*y / r2_5)
    dUxz = (0
            + 3.0*mu1*(x+mu)*z / r1_5
            + 3.0*mu2*(x-1+mu)*z / r2_5)
    
    # partial wrt y
    dUyx = dUxy  # symmetry for second partial derivatives
    dUyy = (1
            - mu1*(1.0/r1_3 - 3.0*y**2 / r1_5)
            - mu2*(1.0/r2_3 - 3.0*y**2 / r2_5))
    dUyz = (0
            + 3.0*mu1*y*z / r1_5
            + 3.0*mu2*y*z / r2_5)
    
    # partial wrt z
    dUzx = dUxz
    dUzy = dUyz
    dUzz = (0
            - mu1*(1.0/r1_3 - 3.0*z**2 / r1_5)
            - mu2*(1.0/r2_3 - 3.0*z**2 / r2_5))
    
    # The velocity coupling in the rotating frame:
    # dx/dt   = vx                 => partial(d x/dt)/partial(vx) = 0 except x->vx
    # dy/dt   = vy                 => partial(d y/dt)/partial(vy) = 0 except y->vy
    # dz/dt   = vz
    # dvx/dt  = ax(...) + 2 vy     => partial wrt vy is 2
    # dvy/dt  = ay(...) - 2 vx     => partial wrt vx is -2
    # dvz/dt  = az(...)
    
    # Construct the 6x6 Jacobian matrix:
    #   [0,  0,  0,  1,  0,  0 ]
    #   [0,  0,  0,  0,  1,  0 ]
    #   [0,  0,  0,  0,  0,  1 ]
    #   [dUxx,dUxy,dUxz,0,  2,  0 ]
    #   [dUyx,dUyy,dUyz,-2, 0,  0 ]
    #   [dUzx,dUzy,dUzz,0,  0,  0 ]
    
    jac = np.array([
        [0,    0,    0,    1,    0,    0],
        [0,    0,    0,    0,    1,    0],
        [0,    0,    0,    0,    0,    1],
        [dUxx, dUxy, dUxz, 0,    2,    0],
        [dUyx, dUyy, dUyz,-2,    0,    0],
        [dUzx, dUzy, dUzz, 0,    0,    0]
    ], dtype=float)
    
    return jac

def variational_equations(t, PHI_vec, mu):
    """
    3D variational equations for the CR3BP.
    
    PHI_vec is a 42-element vector:
       first 6 = [x, y, z, vx, vy, vz]
       next 36 = the flattened 6x6 STM, Phi(t).
    
    Returns d/dt(PHI_vec).
    """
    # Unpack the state and the STM
    x = PHI_vec[0]
    y = PHI_vec[1]
    z = PHI_vec[2]
    vx = PHI_vec[3]
    vy = PHI_vec[4]
    vz = PHI_vec[5]
    
    # The 36 entries of the STM
    phi_flat = PHI_vec[6:]
    phi = phi_flat.reshape((6,6))
    
    # Compute the 6D derivative of the state
    state_dot = crtbp_accel(np.array([x, y, z, vx, vy, vz], dtype=np.float64), mu)
    
    # Compute the 6x6 Jacobian wrt x,y,z,vx,vy,vz
    dfmat = jacobian_crtbp(x, y, z, mu)
    
    # Variational equation: dPhi/dt = dfmat * Phi
    phi_dot = dfmat @ phi
    
    # Flatten phi_dot and put it all together
    dPHI_vec = np.zeros_like(PHI_vec)
    # The first 6 are the time derivatives of the state
    dPHI_vec[:6] = state_dot
    # The next 36 are the flatten of phi_dot
    dPHI_vec[6:] = phi_dot.flatten()
    return dPHI_vec

def compute_stm(x0, mu, tf, **solve_kwargs):
    """
    Integrate the 3D CR3BP plus STM from t=0 to t=tf,
    returning (t_array, state_array, Phi_tf).
    
    x0 is a 6-vector of initial conditions.
    Phi(0) = I_6, the 6x6 identity, so total dimension is 42.
    """
    # Build initial 42-vector: [x0, reshape(I_6)]
    PHI0 = np.zeros(6 + 36)
    PHI0[:6] = x0
    PHI0[6:] = np.eye(6).flatten()
    
    def ode_fun(t, y):
        return variational_equations(t, y, mu)
    
    sol = solve_ivp(ode_fun, [0, tf], PHI0, **solve_kwargs)
    
    # The entire trajectory + stm
    t_array = sol.t
    all_y = sol.y.T  # shape (len(t_array), 42)
    
    # The trajectory portion is columns 0..5
    state_array = all_y[:, :6]
    
    # The final row's last 36 elements = monodromy matrix
    phi_tf_flat = all_y[-1, 6:]
    Phi_tf = phi_tf_flat.reshape((6,6))
    return t_array, state_array, Phi_tf
