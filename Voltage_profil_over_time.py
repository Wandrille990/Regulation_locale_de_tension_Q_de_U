import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
# Parameters
num_buses = 10
time_steps = 24

# Simulated distances along the feeder (in km)
distances = np.linspace(0, 25, 179)  # 0 km to 10 km
# Simulated time points (in hours)
times = np.linspace(0, 24, 96)  # 0 to 24 hours

# Create a meshgrid for distances and times
D, T = np.meshgrid(distances, times)

# Simulate voltage magnitudes (in per unit)
# For demonstration, we'll create a pattern that varies with both distance and time
# Base voltage is 1.0 p.u., with variations added
voltages = pd.read_csv(r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats\res_ctrl\res_bus\vm_pu.csv", sep=';')
voltages = voltages.drop("Unnamed: 0", axis=1)

# Create a 3D surface plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(D, T, voltages, cmap='viridis', edgecolor='none')

# Label axes
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Time (hours)')
ax.set_zlabel('Voltage (p.u.)')
ax.set_title('Voltage Profile Over Distance and Time')

# Add a color bar to indicate voltage magnitude
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()