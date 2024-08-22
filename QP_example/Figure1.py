import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function
def f(x1, x2):
    return 2 * x1**2 + 2 * x1 * x2 + 5 * x2**2 + 3 * x1

z_axis = 10
z_axis_minus = -2
x_axis = 3
y_axis = 3
margin = 2

# Generate data for plotting the function surface
x1 = np.linspace(-x_axis, x_axis, 200)
x2 = np.linspace(-y_axis, y_axis, 200)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

# Cut off function values above 10
Z[Z >= z_axis-margin] = np.nan

# Create the 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the vertical plane x1 + x2 = 1
X1_plane = np.linspace(-x_axis, x_axis, 200)
X2_plane = 1 - X1_plane
Z_plane = np.linspace(z_axis_minus, z_axis-margin, 200)
X1_plane, Z_plane = np.meshgrid(X1_plane, Z_plane)
X2_plane = 1 - X1_plane

ax.plot_surface(X1_plane, X2_plane, Z_plane, color='blue', alpha=0.4)

# Plot the horizontal surface -x1 - x2 <= 0
X1_horz = np.linspace(-x_axis, x_axis, 200)
X2_horz = np.linspace(-y_axis, y_axis, 200)
X1_horz, X2_horz = np.meshgrid(X1_horz, X2_horz)
Z_horz = np.linspace(z_axis_minus, z_axis-margin, 200)
Z_horz = np.full_like(X1_horz, 3.75)  # z = 0 plane

# Mask the area where -X1_horz - X2_horz > 0
mask = (-X1_horz - X2_horz) > 0
Z_horz[mask] = np.nan

ax.plot_surface(X1_horz, X2_horz, Z_horz, color='red', alpha=0.4)

# Plot the vertical surfaces along the edges
Z_vert = np.linspace(z_axis_minus, z_axis, 200)
Z_vert, _ = np.meshgrid(Z_vert, np.ones_like(X1_horz))

ax.plot_surface(X1_horz[:, 0], X2_horz[:, 0], Z_vert, color='yellow', alpha=0.2)  # y = -y_axis edge
ax.plot_surface(X1_horz[:, -1], X2_horz[:, -1], Z_vert, color='yellow', alpha=0.2)  # y = y_axis edge
ax.plot_surface(X1_horz[0, :], X2_horz[0, :], Z_vert, color='yellow', alpha=0.2)  # x = -x_axis edge
ax.plot_surface(X1_horz[-1, :], X2_horz[-1, :], Z_vert, color='yellow', alpha=0.2)  # x = x_axis edge

# Plot the function surface
surf = ax.plot_surface(X1, X2, Z, color='gray', alpha=0.6)  # Set the surface color to green

# Plot the red point and add the label
x_star = 0.5
y_star = 0.5
z_star = f(x_star, y_star)
ax.scatter(x_star, y_star, z_star, color='red', s=50, depthshade=False)
ax.text(x_star+0.1, y_star, z_star + 0.25, 'x*', color='red', fontsize=12, weight='bold')

# Set limits for the axes
ax.set_xlim(-x_axis, x_axis)
ax.set_ylim(-x_axis, x_axis)
ax.set_zlim(z_axis_minus, z_axis)

# Set smaller step size for z-axis ticks
ax.set_zticks(np.arange(z_axis_minus, z_axis, 1))
ax.set_xticks(np.arange(-x_axis, x_axis, 1))
ax.set_yticks(np.arange(-x_axis, x_axis, 1))

# Add color bar which maps values to colors
#fig.colorbar(surf, shrink=0.5, aspect=10)

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'f$(x_1, x_2)$')
#ax.set_title(r'3D plot of f$(x_1, x_2)$ with equality constraint (blue) $x_1 + x_2 = 1$ and inequality constraint (red) $-x_1 - x_2 \leq 0$')


# Add longer 3D arrows for axis
arrow_length_pos = x_axis  # Increase arrow length for positive direction
arrow_length_neg = -x_axis  # Increase arrow length for negative direction
arrow_length_z_minus = z_axis_minus  # Different arrow length for z axis
arrow_length_z_plus = z_axis

ax.quiver(0, 0, 0, arrow_length_pos, 0, 0, color='black', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, arrow_length_neg, 0, 0, color='black', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, arrow_length_pos, 0, color='black', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, arrow_length_neg, 0, color='black', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, arrow_length_z_plus, color='black', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, arrow_length_z_minus, color='black', arrow_length_ratio=0.1)

ax.text(arrow_length_pos, 0, 0, r'$x_1$', color='black')
ax.text(arrow_length_neg, 0, 0, "", color='black')
ax.text(0, arrow_length_pos, 0, r'$x_2$', color='black')
ax.text(0, arrow_length_neg, 0, "", color='black')
ax.text(0, 0, arrow_length_z_plus, r'f$(x_1,x_2)$', color='black')
ax.text(0, 0, arrow_length_z_minus, "", color='black')

# Add gridlines for better visualization
ax.grid(True)

# Add numbers with step size 1 for every direction
for i in range(-x_axis+1, x_axis):
    ax.text(i, 0, 0, str(i), color='black')
    ax.text(0, i, 0, str(i), color='black')
for i in range(z_axis_minus+1, z_axis-1):
    ax.text(0, 0, i, str(i), color='black')

plt.show()
