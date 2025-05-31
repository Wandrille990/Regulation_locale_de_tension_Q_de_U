import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
# Parameters
num_buses = 10
time_steps = 24

# Simulated distances along the feeder (in km)
distances = np.linspace(0, 25, 177)  # 0 km to 10 km
# Simulated time points (in hours)
times = np.linspace(0, 24, 96)  # 0 to 24 hours

# Create a meshgrid for distances and times
D, T = np.meshgrid(distances, times)

# Simulate voltage magnitudes (in per unit)
# For demonstration, we'll create a pattern that varies with both distance and time
# Base voltage is 1.0 p.u., with variations added
voltages = pd.read_csv(r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats\res_ctrl\res_bus\vm_pu.csv", sep=';')
voltages = voltages.drop("Unnamed: 0", axis=1)
voltages = voltages.drop(["58", "318"], axis=1)

# Number of rows and columns
# n_rows, n_cols = res_vm_pu_nc_dist_col.shape
# # Generate a list of distinct colors (1 per column)
# colors = plt.cm.get_cmap('tab10', n_cols).colors  # or use any other colormap
# # Create final color list: repeat each color for each row in the column
# scatter_colors = [color for color in colors for _ in range(n_rows)]

# Create a 3D surface plot
# res_vm_pu_nc_dist_col = res_vm_pu_nc.rename({str(key): str(val) for key, val in dict(voltage_profils_c[bus[0]][0]).items()}, axis=1)
# res_vm_pu_nc_dist_col = res_vm_pu_nc_dist_col.rename({str(key): str(val) for key, val in dict(voltage_profils_c[bus[1]][0]).items()}, axis=1)
# res_vm_pu_nc_dist_col = res_vm_pu_nc_dist_col.drop('58', axis=1)
# res_vm_pu_nc_dist_col = res_vm_pu_nc_dist_col.drop('318', axis=1)

# res_vm_pu_c_dist_col = res_vm_pu_c.rename({str(key): str(val) for key, val in dict(voltage_profils_c[bus[0]][0]).items()}, axis=1)
# res_vm_pu_c_dist_col = res_vm_pu_c_dist_col.rename({str(key): str(val) for key, val in dict(voltage_profils_c[bus[1]][0]).items()}, axis=1)
# res_vm_pu_c_dist_col = res_vm_pu_c_dist_col.drop('318', axis=1)
# res_vm_pu_c_dist_col = res_vm_pu_c_dist_col.drop('58', axis=1)

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.scatter(res_vm_pu_nc_dist_col.index.to_numpy().tolist()*len(res_vm_pu_nc_dist_col.columns),
#                   [eval(dist_col) for dist_col in res_vm_pu_nc_dist_col.columns]*len(res_vm_pu_nc_dist_col.index),
#                   res_vm_pu_nc_dist_col.to_numpy().flatten(), cmap='viridis', c=scatter_colors, edgecolor='none')

# # Number of rows and columns
# n_rows, n_cols = res_vm_pu_nc_dist_col.shape
# # Generate a list of distinct colors (1 per column)
# colors = plt.cm.get_cmap('tab10', n_cols).colors  # or use any other colormap
# # Create final color list: repeat each color for each row in the column
# scatter_colors = [color for color in colors for _ in range(n_rows*2)]

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.scatter(res_vm_pu_c_dist_col.index.to_numpy().tolist()*len(res_vm_pu_c_dist_col.columns)*2,
#                   [eval(dist_col) for dist_col in res_vm_pu_c_dist_col.columns]*len(res_vm_pu_c_dist_col.index)*2,
#                   np.concatenate((res_vm_pu_c_dist_col.to_numpy().flatten(), res_vm_pu_nc_dist_col.to_numpy().flatten())), c=scatter_colors, edgecolor='none')

# Label axes
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Time (hours)')
ax.set_zlabel('Voltage (p.u.)')
ax.set_title('Voltage Profile Over Distance and Time')

# Add a color bar to indicate voltage magnitude
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

print("")