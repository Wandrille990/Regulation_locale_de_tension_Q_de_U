import pandapower as pp
import matplotlib.pyplot as plt

# Create empty network
net = pp.create_empty_network()

# Create buses
bus0 = pp.create_bus(net, vn_kv=20., name="Slack Bus")
bus1 = pp.create_bus(net, vn_kv=20., name="Post 1")
bus2 = pp.create_bus(net, vn_kv=20., name="Industrial Area")
bus3 = pp.create_bus(net, vn_kv=20., name="PV")

# Create external grid (slack)
pp.create_ext_grid(net, bus=bus0, vm_pu=1.0, name="Grid Connection")

# Line type
line_type = "NAYY 4x50 SE"

# Create lines (3x10 km segments)
pp.create_line(net, from_bus=bus0, to_bus=bus1, length_km=10, std_type=line_type, name="Line 1")
pp.create_line(net, from_bus=bus1, to_bus=bus2, length_km=10, std_type=line_type, name="Line 2")
pp.create_line(net, from_bus=bus2, to_bus=bus3, length_km=10, std_type=line_type, name="Line 3")

# Classical distribution loads (e.g., street, home loads)
pp.create_load(net, bus=bus1, p_mw=0.2, q_mvar=0.05, name="Post Load 1")
pp.create_load(net, bus=bus2, p_mw=0.25, q_mvar=0.06, name="Post Load 2")

# Industrial load at the end
pp.create_load(net, bus=bus2, p_mw=1.5, q_mvar=0.3, name="Industrial Load")

# PV generation at industrial site (assume daytime production)
pp.create_sgen(net, bus=bus3, p_mw=0.6, q_mvar=0.0, name="PV Plant")

# Run power flow
pp.runpp(net)

# Get voltage profile
buses = [bus0, bus1, bus2, bus3]
bus_names = [net.bus.at[bus, "name"] for bus in buses]
voltages = [net.res_bus.vm_pu[bus] for bus in buses]
distances = [0, 10, 20, 30]  # in km

# Plot
plt.figure(figsize=(9, 5))
plt.plot(distances, voltages, marker='o', linestyle='-', color='blue')
for i, name in enumerate(bus_names):
    plt.text(distances[i], voltages[i] + 0.001, name, ha='center')

plt.title("Voltage Profile Along the Line with Loads and PV")
plt.xlabel("Distance from Slack Bus (km)")
plt.ylabel("Voltage Magnitude (p.u.)")
plt.ylim(0.95, 1.01)
plt.grid(True)
plt.tight_layout()
plt.show()


print("")