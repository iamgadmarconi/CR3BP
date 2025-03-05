"""
Plotting functions for the CR3BP.

This module provides a collection of functions for visualizing trajectories,
manifolds, and other dynamics in the Circular Restricted Three-Body Problem (CR3BP).

The functions are designed to be used with the `src.dynamics.propagator` module,
which provides functions for propagating trajectories in the CR3BP.

The module includes functions for plotting trajectories in the rotating frame,
inertial frame, and libration points, as well as functions for plotting manifolds
and Poincaré sections.
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

from src.utils.crtbp import to_si_units, _get_angular_velocity, si_time
from src.utils.frames import rotating_to_inertial
from src.dynamics.crtbp import libration_points, hill_region, crtbp_energy
from src.dynamics.propagator import propagate_crtbp


def plot_rotating_frame_trajectories(sol, bodies, system_distance, colors=None, figsize=(10, 8)):
    """
    Plot 3D trajectories of particle and celestial bodies in the rotating frame.
    
    Parameters
    ----------
    sol : scipy.integrate.OdeSolution
        Solution object from the ODE solver containing trajectory data.
    bodies : list
        List of celestial body objects with properties like mass, radius, and name.
    system_distance : float
        Characteristic distance of the system in meters.
    colors : list, optional
        List of colors for different trajectories.
    figsize : tuple, default=(10, 8)
        Figure size in inches (width, height).
        
    Returns
    -------
    None
        Displays the plot using plt.show().
        
    Notes
    -----
    The function converts dimensionless coordinates to SI units and
    plots both the particle trajectory and the primary/secondary bodies as spheres.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert particle trajectory to SI units
    traj_si = np.array([to_si_units(state, bodies[0].mass, bodies[1].mass, system_distance) 
                       for state in sol.y.T])
    x_si = traj_si[:, 0]
    y_si = traj_si[:, 1]
    z_si = traj_si[:, 2]
    
    # Plot converted particle trajectory
    ax.plot(x_si, y_si, z_si, label='Particle', color='red')
    
    # Calculate mass parameter and get ACTUAL system distance from body parameters
    mu = bodies[1].mass / (bodies[0].mass + bodies[1].mass)
    
    # Convert dimensionless positions to SI units
    state_1_si = to_si_units(bodies[1].r_init, bodies[0].mass, bodies[1].mass, system_distance)
    state_0_si = to_si_units(bodies[0].r_init, bodies[0].mass, bodies[1].mass, system_distance)

    # Plot primary as a sphere (assumed to be bodies[0])
    primary_center = np.array([-mu * system_distance, 0, 0])
    primary_radius = bodies[0].radius
    _plot_body(ax, primary_center, primary_radius, 'blue', bodies[0].name)

    # Plot Secondary as a sphere (assumed to be bodies[1])
    secondary_center = np.array([(1 - mu) * system_distance, 0, 0])
    secondary_radius = bodies[1].radius
    _plot_body(ax, secondary_center, secondary_radius, 'grey', bodies[1].name)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Rotating Frame Trajectories')
    _set_axes_equal(ax)
    ax.legend()
    plt.show()

def plot_inertial_frame_trajectories(sol, bodies, system_distance, colors=None, figsize=(10, 8)):
    """
    Plot 3D trajectories in Primary-centered inertial frame.
    
    Parameters
    ----------
    sol : scipy.integrate.OdeSolution
        Solution object from the ODE solver containing trajectory data.
    bodies : list
        List of celestial body objects with properties like mass, radius, and name.
    system_distance : float
        Characteristic distance of the system in meters.
    colors : list, optional
        List of colors for different trajectories.
    figsize : tuple, default=(10, 8)
        Figure size in inches (width, height).
        
    Returns
    -------
    None
        Displays the plot using plt.show().
        
    Notes
    -----
    The function converts rotating frame coordinates to inertial frame coordinates,
    then to SI units, and plots the particle trajectory along with the primary body
    at the origin and the secondary body's orbit and position.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate system parameters
    mu = bodies[1].mass / (bodies[0].mass + bodies[1].mass)
    omega = _get_angular_velocity(bodies[0].mass, bodies[1].mass, system_distance)
    
    # Convert particle trajectory to inertial frame
    traj_inertial = []
    for state, t in zip(sol.y.T, sol.t):
        # Convert rotating frame (dimensionless) to inertial frame (dimensionless)
        state_inertial = rotating_to_inertial(state, t, omega=1, mu=mu)
        # Convert to SI units
        state_si = to_si_units(state_inertial, bodies[0].mass, bodies[1].mass, system_distance)
        traj_inertial.append(state_si)
    
    traj_inertial = np.array(traj_inertial)
    x_si, y_si, z_si = traj_inertial[:, 0], traj_inertial[:, 1], traj_inertial[:, 2]
    
    # Plot particle trajectory
    ax.plot(x_si, y_si, z_si, label='Particle', color='red')
    
    # Plot Primary at origin
    primary_center = np.array([0, 0, 0])
    primary_radius = bodies[0].radius
    _plot_body(ax, primary_center, primary_radius, 'blue', bodies[0].name)
    
    # Plot Secondary's orbit and current position
    theta = sol.t  # Dimensionless time = rotation angle
    secondary_x = system_distance * np.cos(theta)
    secondary_y = system_distance * np.sin(theta)
    secondary_z = np.zeros_like(theta)
    ax.plot(secondary_x, secondary_y, secondary_z, '--', color='grey', alpha=0.5, label=f'{bodies[1].name} Orbit')
    
    # Plot Secondary's final position
    secondary_radius = bodies[1].radius
    secondary_center = np.array([secondary_x[-1], secondary_y[-1], secondary_z[-1]])
    _plot_body(ax, secondary_center, secondary_radius, 'grey', bodies[1].name)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Inertial Frame Trajectories')
    _set_axes_equal(ax)
    ax.legend()
    plt.show()

def animate_trajectories(sol, bodies, system_distance, interval=20, figsize=(14, 6), save=False):
    """
    Create an animated comparison of trajectories in rotating and inertial frames.
    
    Parameters
    ----------
    sol : scipy.integrate.OdeSolution
        Solution object from the ODE solver containing trajectory data.
    bodies : list
        List of celestial body objects with properties like mass, radius, and name.
    system_distance : float
        Characteristic distance of the system in meters.
    interval : int, default=20
        Time interval between animation frames in milliseconds.
    figsize : tuple, default=(14, 6)
        Figure size in inches (width, height).
    save : bool, default=False
        Whether to save the animation as an MP4 file.
        
    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object.
        
    Notes
    -----
    This function creates a side-by-side animation showing the trajectory in both
    rotating and inertial frames, with consistent axis scaling to maintain proper
    proportions. The animation shows the motion of celestial bodies and the particle
    over time, with time displayed in days.
    """
    fig = plt.figure(figsize=figsize)
    ax_rot = fig.add_subplot(121, projection='3d')
    ax_inert = fig.add_subplot(122, projection='3d')
    
    # ------------------------------------------------------------------------
    # 1) PRE-COMPUTE ALL TRAJECTORIES IN SI UNITS
    # ------------------------------------------------------------------------
    
    mu = bodies[1].mass / (bodies[0].mass + bodies[1].mass)
    omega_real = _get_angular_velocity(bodies[0].mass, bodies[1].mass, system_distance)  # rad/s
    t_si = si_time(sol.t, bodies[0].mass, bodies[1].mass, system_distance)               # seconds
    
    # (a) Rotating-frame path, scaled to meters
    traj_rot = np.array([
        to_si_units(s, bodies[0].mass, bodies[1].mass, system_distance)[:3]
        for s in sol.y.T
    ])
    
    # (b) Inertial-frame path, scaled to meters
    traj_inert = []
    for s_dimless, t_dimless in zip(sol.y.T, sol.t):
        t_current_si = si_time(t_dimless, bodies[0].mass, bodies[1].mass, system_distance)
        # Convert rotating->inertial (still dimensionless)
        s_inert_dimless = rotating_to_inertial(s_dimless, t_current_si, omega_real, mu)
        # Convert inertial dimensionless -> meters
        s_inert_si = to_si_units(s_inert_dimless, bodies[0].mass, bodies[1].mass, system_distance)
        traj_inert.append(s_inert_si[:3])
    traj_inert = np.array(traj_inert)
    
    # (c) Secondary's inertial orbit in meters
    secondary_x = system_distance * np.cos(omega_real * t_si)
    secondary_y = system_distance * np.sin(omega_real * t_si)
    secondary_z = np.zeros_like(secondary_x)
    
    # ------------------------------------------------------------------------
    # 2) DETERMINE A SINGLE GLOBAL BOUNDING BOX (FOR EQUAL AXES)
    # ------------------------------------------------------------------------
    # Gather all x,y,z from rotating frame, inertial frame, and Primary/Secondary centers.
    
    # Primary center in rotating frame: (-mu*R, 0, 0)
    primary_rot_center = np.array([-mu*system_distance, 0, 0])
    # Secondary center in rotating frame: ((1-mu)*R, 0, 0)
    secondary_rot_center = np.array([(1.0 - mu)*system_distance, 0, 0])
    
    # Primary center in inertial frame: (0, 0, 0)
    primary_inert_center = np.array([0, 0, 0])
    # We'll track the final bounding box across all frames
    all_x = np.concatenate([
        traj_rot[:,0],           # rotating frame path
        traj_inert[:,0],         # inertial frame path
        secondary_x,             # secondary inertial x
        [primary_rot_center[0], secondary_rot_center[0], primary_inert_center[0]]
    ])
    all_y = np.concatenate([
        traj_rot[:,1],
        traj_inert[:,1],
        secondary_y,
        [primary_rot_center[1], secondary_rot_center[1], primary_inert_center[1]]
    ])
    all_z = np.concatenate([
        traj_rot[:,2],
        traj_inert[:,2],
        secondary_z,
        [primary_rot_center[2], secondary_rot_center[2], primary_inert_center[2]]
    ])
    
    # Add margin to fit Primary/Secondary spheres
    max_sphere = max(bodies[0].radius, bodies[1].radius)
    margin = 1.2 * max_sphere
    
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin
    z_min, z_max = all_z.min() - margin, all_z.max() + margin
    
    # We'll make the axes equal by forcing them all to share the same half-range
    # (the largest dimension among x,y,z).
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range)
    
    # Center coordinates
    x_mid = 0.5*(x_min + x_max)
    y_mid = 0.5*(y_min + y_max)
    z_mid = 0.5*(z_min + z_max)
    # We'll use half of the largest range as our plot "radius"
    half_extent = 0.5 * max_range
    
    # ------------------------------------------------------------------------
    # 3) HELPER FUNCTIONS
    # ------------------------------------------------------------------------
    def _plot_body(ax, center, radius, color, label=None):
        """
        Helper function to plot a celestial body as a sphere.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on.
        center : ndarray, shape (3,)
            The (x, y, z) coordinates of the center of the sphere.
        radius : float
            The radius of the sphere.
        color : str
            The color to use for the sphere.
        label : str, optional
            The label to display near the sphere.
            
        Returns
        -------
        None
            Modifies the axes in-place.

        Notes
        -----
        Creates a 3D sphere representation of a celestial body with the given
        center, radius, and color, and optionally adds a text label.
        """
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        ax.plot_surface(x, y, z, color=color, alpha=0.6)
        if label:
            # Put text near top of sphere
            ax.text(center[0], center[1], center[2] + 1.2*radius, label, color=color)
    
    def _set_equal_3d_axes(ax):
        """
        Force 3D axes to have equal scale and cover [mid - half_extent, mid + half_extent].
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to set the limits on.
            
        Returns
        -------
        None
            Modifies the axes in-place.
            
        """
        ax.set_xlim3d([x_mid - half_extent, x_mid + half_extent])
        ax.set_ylim3d([y_mid - half_extent, y_mid + half_extent])
        ax.set_zlim3d([z_mid - half_extent, z_mid + half_extent])
    
    # ------------------------------------------------------------------------
    # 4) MATPLOTLIB ANIMATION FUNCTIONS
    # ------------------------------------------------------------------------
    def init():
        """
        Initialize the animation.
        
        Returns
        -------
        tuple
            A tuple containing the figure and the axes.
            
        Notes
        -----
        Clears the axes and sets up the labels and limits.
        """
        # Clear and set up
        for ax in (ax_rot, ax_inert):
            ax.clear()
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")
            _set_equal_3d_axes(ax)
        
        ax_rot.set_title("Rotating Frame (SI Distances)")
        ax_inert.set_title("Inertial Frame (Real Time, Real Ω)")
        return fig,
    
    def update(frame):
        """
        Update the animation for each frame.
        
        Parameters
        ----------
        frame : int
            The current frame number.
            
        Returns
        -------
        tuple
            A tuple containing the figure and the axes.

        Notes
        -----
        Updates the plot for the current frame, clearing the axes and
        setting the title and labels.
        """
        # Clear each time
        ax_rot.clear()
        ax_inert.clear()
        
        for ax in (ax_rot, ax_inert):
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")
        
        # Current real time in days
        current_t_days = t_si[frame] / 86400.0
        fig.suptitle(f"Time = {current_t_days:.2f} days")
        
        # ------------------- ROTATING FRAME PLOT ------------------- #
        ax_rot.plot(traj_rot[:frame+1, 0],
                    traj_rot[:frame+1, 1],
                    traj_rot[:frame+1, 2],
                    color='red', label='Particle')
        
        # Primary in rotating frame => center = (-mu*R, 0, 0)
        _plot_body(ax_rot, primary_rot_center, bodies[0].radius, 'blue', bodies[0].name)
        # Secondary => center = ((1-mu)*R, 0, 0)
        _plot_body(ax_rot, secondary_rot_center, bodies[1].radius, 'gray', bodies[1].name)
        
        ax_rot.set_title("Rotating Frame (SI Distances)")
        ax_rot.legend()
        _set_equal_3d_axes(ax_rot)
        
        # ------------------- INERTIAL FRAME PLOT ------------------- #
        ax_inert.plot(traj_inert[:frame+1, 0],
                      traj_inert[:frame+1, 1],
                      traj_inert[:frame+1, 2],
                      color='red', label='Particle')
        
        # Primary at origin
        _plot_body(ax_inert, primary_inert_center, bodies[0].radius, 'blue', bodies[0].name)
        
        # Secondary orbit so far
        ax_inert.plot(secondary_x[:frame+1], secondary_y[:frame+1], secondary_z[:frame+1],
                      '--', color='gray', alpha=0.5, label=f'{bodies[1].name} orbit')
        # Secondary sphere at current location
        secondary_center_now = np.array([secondary_x[frame], secondary_y[frame], secondary_z[frame]])
        _plot_body(ax_inert, secondary_center_now, bodies[1].radius, 'gray', bodies[1].name)
        
        ax_inert.set_title("Inertial Frame (Real Time, Real Ω)")
        ax_inert.legend()
        _set_equal_3d_axes(ax_inert)
        
        return fig,
    
    # ------------------------------------------------------------------------
    # 5) RUN THE ANIMATION
    # ------------------------------------------------------------------------
    total_frames = len(sol.t)
    frames_to_use = range(0, total_frames, 30)  # e.g. step by 5

    ani = animation.FuncAnimation(
        fig, update,
        frames=frames_to_use,
        init_func=init,
        interval=interval,
        blit=False
    )
    if save:
        ani.save('trajectory.mp4', fps=60, dpi=500)
    plt.show()
    plt.close()
    return ani

def plot_libration_points(bodies, mu, system_distance, figsize=(10, 8)):
    """
    Plot the five libration points in the rotating frame with SI units.
    
    Parameters
    ----------
    bodies : list
        List of celestial body objects with properties like mass, radius, and name.
    mu : float
        Mass parameter of the system (ratio of secondary mass to total mass).
    system_distance : float
        Characteristic distance of the system in meters.
    figsize : tuple, default=(10, 8)
        Figure size in inches (width, height).
        
    Returns
    -------
    None
        Displays the plot using plt.show().
        
    Notes
    -----
    Plots the five Lagrangian points (L1-L5) in the rotating frame, along with
    the primary and secondary bodies as spheres.
    """
    ax = _plot_libration_points(mu, system_distance, figsize)
    center_primary_dimless = bodies[0].r_init
    center_secondary_dimless = bodies[1].r_init

    center_primary_si = to_si_units(center_primary_dimless, bodies[0].mass, bodies[1].mass, system_distance)
    center_secondary_si = to_si_units(center_secondary_dimless, bodies[0].mass, bodies[1].mass, system_distance)

    _plot_body(ax, center_primary_si, bodies[0].radius, 'blue', bodies[0].name)
    _plot_body(ax, center_secondary_si, bodies[1].radius, 'gray', bodies[1].name)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Libration Points in Rotating Frame')
    _set_axes_equal(ax)
    ax.legend()
    plt.show()

def plot_zvc(bodies, mu, C, x_range=(-2,2), y_range=(-2,2), n_grid=400):
    """
    Plot the Zero Velocity Curve (ZVC) for a given Jacobi constant.
    
    Parameters
    ----------
    bodies : list
        List of celestial body objects with properties like mass, radius, and name.
    mu : float
        Mass parameter of the system (ratio of secondary mass to total mass).
    C : float
        Jacobi constant value for which to plot the ZVC.
    x_range : tuple, default=(-2, 2)
        Range of x values to plot.
    y_range : tuple, default=(-2, 2)
        Range of y values to plot.
    n_grid : int, default=400
        Number of grid points in each dimension.
        
    Returns
    -------
    None
        Displays the plot using plt.show().
        
    Notes
    -----
    Visualizes the Zero Velocity Curve (ZVC) in the x-y plane for a given 
    Jacobi constant. The ZVC separates regions where motion is possible from
    those where it is not.
    """
    X, Y, Z = hill_region(mu, C, x_range, y_range, n_grid)
    plt.figure(figsize=(6, 5))
    cs = plt.contour(X, Y, Z, levels=[0], colors='k')
    plt.clabel(cs, inline=True, fontsize=8, fmt="ZVC")

    plt.contourf(X, Y, Z, levels=[-1e9, 0], alpha=0.5, cmap='RdBu_r')

    plt.plot([-mu], [0], 'o', label=bodies[0].name, color='blue')      # 'bo' means blue circle
    plt.plot([1 - mu], [0], 'o', label=bodies[1].name, color='gray')   # 'ko' means black circle

    plt.gca().set_aspect('equal')
    plt.xlim(*x_range)
    plt.ylim(*y_range)
    plt.title(f"ZVC for mu={mu}, C={C}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

def plot_orbit_family(xL, t1L, mu):
    """
    Plot a family of periodic orbits around a libration point.
    
    Parameters
    ----------
    xL : ndarray, shape (N, 6)
        Array of initial states for each orbit in the family.
    t1L : ndarray, shape (N,)
        Array of half-periods for each orbit.
    mu : float
        Mass parameter of the system (ratio of secondary mass to total mass).
        
    Returns
    -------
    None
        Displays the plot using plt.show().
        
    Notes
    -----
    For each initial state and half-period, propagates a full orbit using the
    CRTBP equations and plots the resulting 3D trajectory.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Use tqdm for progress bar
    for (x0, t1) in tqdm(zip(xL, t1L), total=len(xL), desc="Plotting orbits"):
        T = 2.0 * t1
        sol = propagate_crtbp(x0, mu, T)
        ts = sol.t
        Xs = sol.y.T
        ax.plot(Xs[:, 0], Xs[:, 1], Xs[:, 2])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    _set_axes_equal(ax)
    plt.show()

def plot_orbit_family_energy(xL, t1L, mu, xL_i):
    """
    Plot a family of periodic orbits with their energy difference from a libration point.
    
    Parameters
    ----------
    xL : ndarray, shape (N, 6)
        Array of initial states for each orbit in the family.
    t1L : ndarray, shape (N,)
        Array of half-periods for each orbit.
    mu : float
        Mass parameter of the system (ratio of secondary mass to total mass).
    xL_i : ndarray, shape (6,)
        State vector of the libration point to use as reference for energy.
        
    Returns
    -------
    None
        Displays the plot using plt.show().
        
    Notes
    -----
    For each orbit in the family, computes the energy relative to the libration point
    and plots the orbit in (x, y, dE) space, where dE is the energy difference.
    """
    E_L = crtbp_energy([xL_i[0], xL_i[1], xL_i[2], 0, 0, 0], mu)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for (x0, t1) in tqdm(zip(xL, t1L), total=len(xL), desc="Plotting orbits"):
        T = 2.0 * t1
        sol = propagate_crtbp(x0, mu, T)
        ts = sol.t
        Xs = sol.y.T
        E_vals = np.array([crtbp_energy(X, mu) for X in Xs])
        E_diff = E_vals - E_L
        xs = Xs[:, 0]
        ys = Xs[:, 1]
        ax.plot(xs, ys, E_diff)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(f'E - Libration Point Energy')
    ax.set_title('Family of Orbits vs. Energy above Libration Point')
    _set_axes_equal(ax)
    plt.show()

def plot_manifold(bodies, xW_list, tW_list, system_distance):
    """
    Plot the manifold structure of a periodic orbit.
    
    Parameters
    ----------
    bodies : list
        List of celestial body objects with properties like mass, radius, and name.
    xW_list : list of ndarray
        List of state trajectories for each manifold branch.
    tW_list : list of ndarray
        List of time arrays corresponding to each manifold trajectory.
    system_distance : float
        Characteristic distance of the system in meters.
        
    Returns
    -------
    None
        Displays the plot using plt.show().
        
    Notes
    -----
    Plots the manifold structure (stable/unstable manifolds) of a periodic orbit
    in the rotating frame, along with the primary and secondary bodies.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for xW, tW in zip(xW_list, tW_list):
        ax.plot(xW[:, 0], xW[:, 1], xW[:, 2], 'b-')
    
    mu = bodies[1].mass / (bodies[0].mass + bodies[1].mass)

    # Plot primary as a sphere (assumed to be bodies[0])
    primary_center = np.array([-mu, 0, 0])
    primary_radius = bodies[0].radius
    _plot_body(ax, primary_center, primary_radius / system_distance, 'blue', bodies[0].name)

    # Plot Secondary as a sphere (assumed to be bodies[1])
    secondary_center = np.array([(1 - mu), 0, 0])
    secondary_radius = bodies[1].radius
    _plot_body(ax, secondary_center, secondary_radius / system_distance, 'grey', bodies[1].name)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    _set_axes_equal(ax)
    ax.set_title('Manifold')
    plt.show()

def plot_poincare_section(ysos, ydsos, system_distance):
    """
    Plot the Poincaré section of a trajectory.
    
    Parameters
    ----------
    ysos : ndarray
        Array of y-coordinates at the section crossings.
    ydsos : ndarray
        Array of y-velocities at the section crossings.
    system_distance : float
        Characteristic distance of the system in meters.
        
    Returns
    -------
    None
        Displays the plot using plt.show().
        
    Notes
    -----
    Creates a Poincaré section plot showing y vs. y-dot at the crossings,
    which helps visualize the dynamics and detect chaotic or regular behavior.
    """
    ysos = np.append(ysos, ysos[0])
    ydsos = np.append(ydsos, ydsos[0])

    # Create the plot
    plt.figure()
    plt.plot(ysos, ydsos, 'b-', linewidth=5)
    plt.title('Poincare section')
    plt.xlabel('$y$')
    plt.ylabel(r'$\dot{y}$')
    plt.grid(True)
    plt.show()

def _plot_libration_points(mu, system_distance, figsize=(10, 8)):
    """
    Helper function to plot the five libration points in the rotating frame.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the system (ratio of secondary mass to total mass).
    system_distance : float
        Characteristic distance of the system in meters.
    figsize : tuple, default=(10, 8)
        Figure size in inches (width, height).
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.
        
    Notes
    -----
    Creates a plot with the five Lagrangian points (L1-L5) marked using different
    colors and markers. This is a helper function used by plot_libration_points.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get all libration points in dimensionless coordinates
    collinear, equilateral = libration_points(mu)
    all_points = collinear + equilateral  # Combine L1-3 and L4-5
    
    # Convert to SI units and plot with different markers
    labels = ['L1', 'L2', 'L3', 'L4', 'L5']
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    markers = ['o', '^', 's', 'D', 'X']
    
    for point, label, color, marker in zip(all_points, labels, colors, markers):
        x, y, z = point * system_distance  # Convert to SI units
        ax.scatter(x, y, z, color=color, marker=marker, s=50, label=label)

    return ax

def _plot_body(ax, center, radius, color, label=None):
    """
    Helper function to plot a celestial body as a sphere.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    center : ndarray, shape (3,)
        The (x, y, z) coordinates of the center of the sphere.
    radius : float
        The radius of the sphere.
    color : str
        The color to use for the sphere.
    label : str, optional
        The label to display near the sphere.
        
    Returns
    -------
    None
        Modifies the axes in-place.
        
    Notes
    -----
    Creates a 3D sphere representation of a celestial body with the given
    center, radius, and color, and optionally adds a text label.
    """
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=0.6)
    if label:
        # Put text near top of sphere
        ax.text(center[0], center[1], center[2] + 1.2*radius, label, color=color)

def _set_axes_equal(ax):
    """
    Make the 3D axes have equal scale so that spheres look like spheres.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to adjust.
        
    Returns
    -------
    None
        Modifies the axes in-place.
        
    Notes
    -----
    Adjusts the limits of the 3D axes to ensure that the scale is equal in all
    dimensions, which prevents distortion of spherical objects.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = 0.5 * max([x_range, y_range, z_range])
    
    x_mid = 0.5 * sum(x_limits)
    y_mid = 0.5 * sum(y_limits)
    z_mid = 0.5 * sum(z_limits)
    
    ax.set_xlim3d([x_mid - max_range, x_mid + max_range])
    ax.set_ylim3d([y_mid - max_range, y_mid + max_range])
    ax.set_zlim3d([z_mid - max_range, z_mid + max_range])
