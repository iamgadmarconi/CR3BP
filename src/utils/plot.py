import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from utils.crtbp import to_si_units

def plot_trajectories(sol, bodies, system_distance, colors=None, figsize=(10, 8)):
    """
    Plot 3D trajectories of satellite and celestial bodies with spheres for bodies.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert spacecraft trajectory to SI units
    traj_si = np.array([to_si_units(state, bodies[0].mass, bodies[1].mass, system_distance) 
                       for state in sol.y.T])
    x_si = traj_si[:, 0]
    y_si = traj_si[:, 1]
    z_si = traj_si[:, 2]
    
    # Plot converted satellite trajectory
    ax.plot(x_si, y_si, z_si, label='Spacecraft', color='red')
    
    # Calculate mass parameter and get ACTUAL system distance from body parameters
    mu = bodies[1].mass / (bodies[0].mass + bodies[1].mass)
    
    # Convert dimensionless positions to SI units
    state_1_si = to_si_units(bodies[1].r_init, bodies[0].mass, bodies[1].mass, system_distance)
    state_0_si = to_si_units(bodies[0].r_init, bodies[0].mass, bodies[1].mass, system_distance)

    # Define grid for sphere surface (u: azimuthal, v: polar angles)
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    
    # Plot Earth as a sphere (assumed to be bodies[0])
    primary_center = np.array([-mu * system_distance, 0, 0])
    primary_radius = bodies[0].radius
    primary_x = primary_radius * np.cos(u) * np.sin(v) + primary_center[0]
    primary_y = primary_radius * np.sin(u) * np.sin(v) + primary_center[1]
    primary_z = primary_radius * np.cos(v) + primary_center[2]
    ax.plot_surface(primary_x, primary_y, primary_z, color='blue', alpha=0.6, label=bodies[0].name)
    
    # Plot Moon as a sphere (assumed to be bodies[1])
    secondary_center = np.array([(1 - mu) * system_distance, 0, 0])
    secondary_radius = bodies[1].radius
    secondary_x = secondary_radius * np.cos(u) * np.sin(v) + secondary_center[0]
    secondary_y = secondary_radius * np.sin(u) * np.sin(v) + secondary_center[1]
    secondary_z = secondary_radius * np.cos(v) + secondary_center[2]
    ax.plot_surface(secondary_x, secondary_y, secondary_z, color='grey', alpha=0.6, label=bodies[1].name)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('System Trajectories')
    _set_axes_equal(ax)
    ax.legend()
    plt.show()


def _set_axes_equal(ax):
    """
    Set 3D plot axes to equal scale so spheres appear as spheres.
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
