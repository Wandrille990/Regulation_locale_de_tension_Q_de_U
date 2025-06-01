import pandapower.networks as pn
import pandapower.topology as top
from pandapower.control import ConstControl
import pandapower.control.controller.DERController as DERModels
from pandapower.timeseries import run_timeseries, OutputWriter
from pandapower.timeseries.data_sources.frame_data import DFData

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple


# 1. Load existing network (e.g., MV network with PV)
net_0 = pn.mv_oberrhein(scenario="generation")

# 2. Create a high-PV / low-load timeseries
time_steps = 24*4
pv_scaling = np.clip(np.sin(np.linspace(0, np.pi, time_steps)), 0.1, 1.0)  # Peaks at midday
pv_scaling_matrix = np.tile(pv_scaling.reshape(-1, 1), (1, len(net_0.sgen.index)))
df_pgen = net_0.sgen.p_mw.values * pd.DataFrame(pv_scaling_matrix, index=list(range(time_steps)), columns=net_0.sgen.index)
ds_data_pgen = DFData(df_pgen)

load_scaling = np.clip(0.3 + 0.2 * np.cos(np.linspace(0, np.pi, time_steps)+np.pi), 0.2, 0.5)  # Low at midday
load_scaling_matrix = np.tile(load_scaling.reshape(-1, 1), (1, len(net_0.load.index)))
df_load = net_0.load.p_mw.values * pd.DataFrame(load_scaling_matrix, index=list(range(time_steps)), columns=net_0.load.index)
ds_data_load = DFData(df_load)

def PQ_area():
    p_deadzone_threshold = 0.05
    arc_angle_limit = np.pi / 7
    num_arc_points = 10*10
    q_deadzone_amplitude = 1e-7

    # Step 1: Dead zone vertical bar (Q = ±1e-7)
    p_deadzone = np.linspace(0, p_deadzone_threshold, 10)
    q_deadzone_top = np.full_like(p_deadzone, q_deadzone_amplitude)
    q_deadzone_bottom = np.full_like(p_deadzone, -q_deadzone_amplitude)

    # Step 2: Top arc: θ in [-π/7, π/7], constrained to P ≥ 0.05
    theta_arc = np.linspace(-arc_angle_limit, arc_angle_limit, num_arc_points)
    p_arc = np.cos(theta_arc)
    q_arc = np.sin(theta_arc)

    # Filter arc to P ≥ threshold
    mask = p_arc >= p_deadzone_threshold
    p_arc_valid = p_arc[mask]
    q_arc_valid = q_arc[mask]

    # Step 3: Construct shape by concatenating segments
    p_shape = np.concatenate([
        p_deadzone,  # Up along left side of rectangle
        p_arc_valid,  # Along circular arc (top)
        p_deadzone[::-1],  # Down along right side of rectangle
        [p_deadzone[0]]  # Close back to starting point
    ])
    q_shape = np.concatenate([
        q_deadzone_bottom,  # Q = -1e-7
        q_arc_valid,  # Top arc
        q_deadzone_top[::-1],  # Q = +1e-7
        [q_deadzone_bottom[0]]  # Closing point
    ])

    return p_shape, q_shape


def opt_run_timeseries(net, ds_gen, ds_load, output_path, control_enabled=False):
    # initialising the outputwriter to save data to excel files in the current folder. You can change this to .json, .csv, or .pickle as well
    ow = OutputWriter(net, output_path=output_path, output_file_type=".csv")
    # adding vm_pu of all buses and line_loading in percent of all lines as outputs to be stored
    ow.log_variable('sgen', 'p_mw')
    ow.log_variable('sgen', 'q_mvar')
    ow.log_variable('sgen', 'sn_mva')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'p_mw')
    ow.log_variable('res_bus', 'q_mvar')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_sgen', 'p_mw')
    ow.log_variable('res_sgen', 'q_mvar')

    const_pload_ctrl = ConstControl(net, element_index=net.load.index, element='load', data_source=ds_load, variable='p_mw',
                                    profile_name=ds_load.df.columns, level=1, order=1)
    const_psgen_ctrl = ConstControl(net, element_index=net.sgen.index, element='sgen', data_source=ds_gen, variable='p_mw',
                                    profile_name=ds_gen.df.columns, level=1, order=2)

    if control_enabled:
        vm_points_pu = np.array([0, 0.96, 0.9725, 1.0375, 1.05, 2])  # DTR Enedis-NOI-RES_60E
        cosphi_points = np.array([0.9285, 0.9285, 0, 0, -0.9285, -0.9285])  # tan(phi) = jusqu'à 0,4 en sous tension, jusqu'à -0,4 en sur tension
        der_ctrl = DERModels.DERController(net, element_index=net.sgen.index, element='sgen', data_source=ds_gen,
                                           p_profile=ds_gen.df.columns, saturate_sn_mva=1,
                                           q_model=DERModels.QModelCosphiVCurve(DERModels.CosphiVCurve(vm_points_pu, cosphi_points)),
                                           # pqv_area=DERModels.PQAreaPOLYGON(*PQ_area()),
                                           level=2)

    run_timeseries(net)

# 3. Initialise networks
net_no_control = pn.mv_oberrhein(scenario="generation")
net_no_control.sgen.scaling = 1
net_control = pn.mv_oberrhein(scenario="generation")
net_control.sgen.scaling = 1

# Simule une situation de sur tension intrinsèque en butée basse rapport de transformation
for trsf_id in net_control.trafo.index:
    net_control.trafo.loc[trsf_id, "tap_pos"] = -5  # net_control.trafo.loc[trsf_id, "tap_min"]
for trsf_id in net_no_control.trafo.index:
    net_no_control.trafo.loc[trsf_id, "tap_pos"] = -5  # net_no_control.trafo.loc[trsf_id, "tap_min"]


# 4. Run time series simulations
output_path_no_ctrl = r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats\res_no_ctrl"
output_path_ctrl = r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats\res_ctrl"

opt_run_timeseries(net_no_control, ds_data_pgen, ds_data_load, output_path_no_ctrl, control_enabled=False)
opt_run_timeseries(net_control, ds_data_pgen, ds_data_load, output_path_ctrl, control_enabled=True)


# 5. Get and analyse results
res_vm_pu_nc = pd.read_csv(r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats\res_no_ctrl\res_bus\vm_pu.csv", sep=';').drop("Unnamed: 0", axis=1)
res_vm_pu_c = pd.read_csv(r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats\res_ctrl\res_bus\vm_pu.csv", sep=';').drop("Unnamed: 0", axis=1)

res_p_mw_nc = pd.read_csv(r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats\res_no_ctrl\res_sgen\p_mw.csv", sep=';').drop("Unnamed: 0", axis=1)
res_p_mw_c = pd.read_csv(r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats\res_ctrl\res_sgen\p_mw.csv", sep=';').drop("Unnamed: 0", axis=1)

print(f"Energie totale produite sur la journée sans contrôle : {round(res_p_mw_nc.sum().sum()*15/60, 3)} MWh")
print(f"Energie totale produite sur la journée avec contrôle : {round(res_p_mw_c.sum().sum()*15/60, 3)} MWh")

def get_voltage_profil(net, res_vm_pu, time):

    # Retrieve the indices of external grid buses
    ext_grid_buses = net.ext_grid.bus.tolist()

    # Create a graph representation of the network
    mg = top.create_nxgraph(net, respect_switches=True)

    # Dictionary to store buses supplied by each external grid
    supplied_buses = {}

    for bus in ext_grid_buses:
        # Find all buses connected to the external grid bus
        connected = top.connected_component(mg, bus)
        supplied_buses[bus] = list(connected)[1:]

    sorted_dist_volt = {}
    ordered_dist_volt = {}
    Profile_sorted = namedtuple('Profile_sorted', ['distances', 't_voltages'])
    Profile_ordered = namedtuple('Profile_ordered', ['distances', 'all_voltages'])
    distances_from_bus = None
    all_voltages_of_bus = None

    for source_bus, buses in supplied_buses.items():
        # Calculate distances from the source bus
        distances_from_bus = top.calc_distance_to_bus(net, source_bus).loc[buses]

        # Extract voltage magnitudes for the buses
        voltages = res_vm_pu.loc[time, [str(bus) for bus in buses]]
        all_voltages_of_bus = res_vm_pu.loc[:, [str(bus) for bus in buses]]

        # Sort distances and corresponding voltages for plotting
        sorted_indices = distances_from_bus.sort_values().index
        sorted_distances = distances_from_bus.loc[sorted_indices]
        sorted_voltages = voltages.loc[[str(id) for id in sorted_indices]]
        # sorted_all_voltages_of_bus = voltages.loc[:,[str(id) for id in sorted_indices]]
        sorted_dist_volt[source_bus] = Profile_sorted(distances=sorted_distances, t_voltages=sorted_voltages)
        ordered_dist_volt[source_bus] = Profile_ordered(distances=distances_from_bus, all_voltages=all_voltages_of_bus)

    return sorted_dist_volt, ordered_dist_volt

midday = time_steps // 2
voltage_profils_nc_48, voltage_profils_nc = get_voltage_profil(net_no_control, res_vm_pu_nc, midday)
voltage_profils_c_48, voltage_profils_c = get_voltage_profil(net_control, res_vm_pu_c, midday)

voltage_profils_nc_90, _ = get_voltage_profil(net_no_control, res_vm_pu_nc, 90)
voltage_profils_c_90, _ = get_voltage_profil(net_control, res_vm_pu_c, 90)


bus = list(voltage_profils_nc_48.keys())

plt.figure(figsize=(10, 6))

plt.scatter(voltage_profils_nc_48[bus[0]].distances, voltage_profils_nc_48[bus[0]].t_voltages,
            marker='o', color='tab:orange', label=f'Feeder from Bus {bus[0]}, without control, 12h')
# plt.scatter(voltage_profils_nc_48[bus[1]].distance, voltage_profils_c[bus[1]].voltage,
#             marker='o', color='tab:orange', label=f'Feeder from Bus {bus[1]}, without control, 12h')
plt.scatter(voltage_profils_nc_90[bus[0]].distances, voltage_profils_nc_90[bus[0]].t_voltages,
            marker='o', color='tab:red', label=f'Feeder from Bus {bus[0]}, without control, 22h30')

plt.scatter(voltage_profils_c_48[bus[0]].distances, voltage_profils_c_48[bus[0]].t_voltages,
            marker='o', color='tab:cyan', label=f'Feeder from Bus {bus[0]}, with control, 12h')
# plt.scatter(voltage_profils_c_48[bus[1]].distance, voltage_profils_nc[bus[1]].voltage,
#             marker='o', color='tab:blue', label=f'Feeder from Bus {bus[1]}, with control, 12h')
plt.scatter(voltage_profils_c_90[bus[0]].distances, voltage_profils_c_90[bus[0]].t_voltages,
            marker='o', color='tab:blue', label=f'Feeder from Bus {bus[0]}, with control, 22h30')

plt.xlabel('Distance from Source Bus (km)')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.title(f'Voltage Profile Along Feeder from Bus {bus}')
plt.grid(True)
plt.legend()
plt.show()


# Plot voltage profile at midday (highest PV, lowest load)
# Tension de tous les bus à un 1/4 d'heure de la journée
# midday = time_steps // 2
# plt.figure(figsize=(12, 6))
# plt.plot(range(results_nc.shape[1]), results_nc.iloc[midday, :], 'ro-', label="No Q=f(U) Control")
# plt.plot(range(results_c.shape[1]), results_c.iloc[midday, :], 'g.-', label="With Q=f(U) Control")
# plt.axhline(1.1, color='k', linestyle='--', label="Voltage Limits (±10%)")
# plt.axhline(0.9, color='k', linestyle='--')
# plt.title("Voltage Profile at Midday (High PV, Low Load)")
# plt.xlabel("Bus Index")
# plt.ylabel("Voltage [p.u.]")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# Tension d'un bus sur la journée
# id_bus = 121
# plt.figure(figsize=(12, 6))
# plt.plot(range(results_nc.shape[0]), results_nc.iloc[:, id_bus], 'ro-', label="No Q=f(U) Control")
# plt.plot(range(results_c.shape[0]), results_c.iloc[:, id_bus], 'g.-', label="With Q=f(U) Control")
# plt.axhline(1.1, color='k', linestyle='--', label="Voltage Limits (±10%)")
# plt.axhline(0.9, color='k', linestyle='--')
# plt.title("Voltage Profile at Midday (High PV, Low Load)")
# plt.xlabel("Bus Index")
# plt.ylabel("Voltage [p.u.]")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

print("")