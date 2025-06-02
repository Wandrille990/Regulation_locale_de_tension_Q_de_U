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


def create_load_prod_profils(net, time_steps=24*4):

    # Define solar window (e.g., 6 AM to 8 PM)
    sunrise_step = 28  # 7 AM
    sunset_step = 82  # 8.5 PM

    # Half sine wave for solar production
    solar_profile = np.zeros(time_steps)
    # Generate sine wave only between sunrise and sunset
    daylight_hours = np.arange(sunrise_step+1, sunset_step)
    # Scale to pi: full half sine wave
    solar_profile[daylight_hours] = np.sin((daylight_hours - sunrise_step) / (sunset_step - sunrise_step) * np.pi)
    # Normalize and scale to max output
    solar_profile = solar_profile / solar_profile.max()

    pv_scaling_matrix = np.tile(solar_profile.reshape(-1, 1), (1, len(net.sgen.index)))
    df_pgen = net.sgen.p_mw.values * pd.DataFrame(pv_scaling_matrix, index=list(range(time_steps)), columns=net.sgen.index)
    ds_data_pgen = DFData(df_pgen)

    # Define Load
    time = np.arange(time_steps)
    hours = time / 4  # Convert steps to hour of day

    # Base load (e.g., night-time low)
    base = 2
    # Smooth increase from 5 AM to 12 PM
    morning_rise = 0.4 * np.clip((hours - 4.5) / 7, 0, 1)
    # Broad midday peak centered at 12 PM
    midday_peak = 0.5 * np.exp(-0.55 * ((hours - 10) / 3)**2)
    # Evening peak centered at 19:00 (7 PM), sharper
    evening_peak = 0.5 * np.exp(-0.5 * ((hours - 19) / 1.3)**2)  # increased to make it the highest
    # Decline after 9 PM
    night_decline = np.where(hours >= 21, -0.05 * (hours - 21), 0)
    # Create a smooth curve that decreases from 0.2 (at 0h) to 0 (at 5h)
    midnight_decline = np.zeros_like(hours)
    midnight_decline_mask = (hours >= 0) & (hours < 4.5)
    midnight_decline[midnight_decline_mask] = 0.27 * (1 - (hours[midnight_decline_mask] / 4.5))  # linear decline
    # Combine all components
    load_profile = base + morning_rise + midday_peak + evening_peak + night_decline + midnight_decline
    load_profile_normalized = load_profile / load_profile.max()

    load_scaling_matrix = np.tile(load_profile_normalized.reshape(-1, 1), (1, len(net.load.index)))
    df_pload = net.load.p_mw.values * pd.DataFrame(load_scaling_matrix, index=list(range(time_steps)), columns=net.load.index)
    df_qload = net.load.q_mvar.values * pd.DataFrame(load_scaling_matrix, index=list(range(time_steps)), columns=net.load.index)

    ds_data_pload = DFData(df_pload)
    ds_data_qload = DFData(df_qload)

    return ds_data_pgen, ds_data_pload, ds_data_qload

def PQ_area(p_deadzone_threshold = 0.05, arc_angle_limit = np.pi / 7, num_arc_points = 10*10, q_deadzone_amplitude = 1e-7):

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

    # plt.figure(figsize=(6,6))
    # plt.plot(q_shape, p_shape, label="PQ Capability", linewidth=2)
    # plt.xlabel("Q (pu)", fontsize=16)
    # plt.ylabel("P (pu)", fontsize=16)
    # plt.title("PQ Diagram with sqrt(P²+Q²) <= 1 and dead zone for P <= 0.05 pu", fontsize=18)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.gca().set_aspect('equal')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    return p_shape, q_shape


def opt_run_timeseries(net, ds_gen, ds_pload, ds_qload, output_path, control_enabled=False):
    # initialising the outputwriter to save data to excel files in the current folder. You can change this to .json, .csv, or .pickle as well
    ow = OutputWriter(net, output_path=output_path, output_file_type=".csv")
    # adding vm_pu of all buses and line_loading in percent of all lines as outputs to be stored
    ow.log_variable('sgen', 'p_mw')
    ow.log_variable('sgen', 'q_mvar')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'p_mw')
    ow.log_variable('res_bus', 'q_mvar')
    ow.log_variable('res_line', 'pl_mw')
    ow.log_variable('res_line', 'ql_mvar')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_sgen', 'p_mw')
    ow.log_variable('res_sgen', 'q_mvar')

    const_pload_ctrl = ConstControl(net, element_index=net.load.index, element='load', data_source=ds_pload, variable='p_mw',
                                    profile_name=ds_pload.df.columns, level=1, order=1)
    const_qload_ctrl = ConstControl(net, element_index=net.load.index, element='load', data_source=ds_qload, variable='q_mvar',
                                    profile_name=ds_qload.df.columns, level=1, order=2)
    const_psgen_ctrl = ConstControl(net, element_index=net.sgen.index, element='sgen', data_source=ds_gen, variable='p_mw',
                                    profile_name=ds_gen.df.columns, level=1, order=3)

    if control_enabled:
        vm_points_pu = np.array([0, 0.96, 0.9725, 1.0375, 1.05, 2])  # DTR Enedis-NOI-RES_60E
        cosphi_points = np.array([0.9285, 0.9285, 1, 1, -0.9285, -0.9285])  # tan(phi) jusqu'à 0,4 en sous tension, jusqu'à -0,4 en sur tension
        der_ctrl = DERModels.DERController(net, element_index=net.sgen.index, element='sgen', data_source=ds_gen,
                                           p_profile=ds_gen.df.columns, saturate_sn_mva=1,
                                           q_model=DERModels.QModelCosphiVCurve(DERModels.CosphiVCurve(vm_points_pu, cosphi_points)),
                                           # pqv_area=DERModels.PQAreaPOLYGON(*PQ_area()),
                                           level=2)

    run_timeseries(net)

def get_voltage_profil(net, res_vm_pu, time_step):

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
        voltages = res_vm_pu.loc[time_step, [str(bus) for bus in buses]]
        all_voltages_of_bus = res_vm_pu.loc[:, [str(bus) for bus in buses]]

        # Sort distances and corresponding voltages for plotting
        sorted_indices = distances_from_bus.sort_values().index
        sorted_distances = distances_from_bus.loc[sorted_indices]
        sorted_voltages = voltages.loc[[str(id) for id in sorted_indices]]
        # sorted_all_voltages_of_bus = voltages.loc[:,[str(id) for id in sorted_indices]]
        sorted_dist_volt[source_bus] = Profile_sorted(distances=sorted_distances, t_voltages=sorted_voltages)
        ordered_dist_volt[source_bus] = Profile_ordered(distances=distances_from_bus, all_voltages=all_voltages_of_bus)

    return sorted_dist_volt, ordered_dist_volt


# 1. Initialise networks
net_no_control = pn.mv_oberrhein(scenario="generation")
net_no_control.sgen.scaling = 1
net_control = pn.mv_oberrhein(scenario="generation")
net_control.sgen.scaling = 1

# Simule une situation de sur tension intrinsèque en butée basse rapport de transformation
for trsf_id in net_control.trafo.index:
    net_control.trafo.loc[trsf_id, "tap_pos"] = -5  # net_control.trafo.loc[trsf_id, "tap_min"]
for trsf_id in net_no_control.trafo.index:
    net_no_control.trafo.loc[trsf_id, "tap_pos"] = -5  # net_no_control.trafo.loc[trsf_id, "tap_min"]


# 2. Run time series simulations
base_res_path = r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats"

output_path_no_ctrl = base_res_path + r"\res_no_ctrl"
output_path_ctrl = base_res_path + r"\res_ctrl"

time_steps = 24*4
ds_data_pgen, ds_data_pload, ds_data_qload = create_load_prod_profils(net_no_control, time_steps)

opt_run_timeseries(net_no_control, ds_data_pgen, ds_data_pload, ds_data_qload, output_path_no_ctrl, control_enabled=False)
opt_run_timeseries(net_control, ds_data_pgen, ds_data_pload, ds_data_qload, output_path_ctrl, control_enabled=True)


# 3. Get and analyse results
res_p_mw_nc = pd.read_csv(base_res_path + r"\res_no_ctrl\res_sgen\p_mw.csv", sep=';').drop("Unnamed: 0", axis=1)
res_p_mw_c = pd.read_csv(base_res_path + r"\res_ctrl\res_sgen\p_mw.csv", sep=';').drop("Unnamed: 0", axis=1)

print(f"Energie totale produite sur la journée sans contrôle : {round(res_p_mw_nc.sum().sum()*15/60, 3)} MWh")
print(f"Energie totale produite sur la journée avec contrôle : {round(res_p_mw_c.sum().sum()*15/60, 3)} MWh")

res_vm_pu_nc = pd.read_csv(base_res_path + r"\res_no_ctrl\res_bus\vm_pu.csv", sep=';').drop("Unnamed: 0", axis=1)
res_vm_pu_c = pd.read_csv(base_res_path + r"\res_ctrl\res_bus\vm_pu.csv", sep=';').drop("Unnamed: 0", axis=1)

midday = time_steps // 2
voltage_profils_nc_48, voltage_profils_nc = get_voltage_profil(net_no_control, res_vm_pu_nc, midday)
voltage_profils_c_48, voltage_profils_c = get_voltage_profil(net_control, res_vm_pu_c, midday)

voltage_profils_nc_80, _ = get_voltage_profil(net_no_control, res_vm_pu_nc, 80)
voltage_profils_c_80, _ = get_voltage_profil(net_control, res_vm_pu_c, 80)


bus = list(voltage_profils_nc_48.keys())

plt.figure(figsize=(10, 6))

plt.scatter(voltage_profils_nc_48[bus[1]].distances, voltage_profils_nc_48[bus[1]].t_voltages,
            marker='o', color='tab:orange', label=f'Feeder from Bus {bus[1]}, without control, 12h')
plt.scatter(voltage_profils_nc_80[bus[1]].distances, voltage_profils_nc_80[bus[1]].t_voltages,
            marker='o', color='tab:red', label=f'Feeder from Bus {bus[1]}, without control, 20h')

plt.scatter(voltage_profils_c_48[bus[1]].distances, voltage_profils_c_48[bus[1]].t_voltages,
            marker='o', color='tab:cyan', label=f'Feeder from Bus {bus[1]}, with control, 12h')
plt.scatter(voltage_profils_c_80[bus[1]].distances, voltage_profils_c_80[bus[1]].t_voltages,
            marker='o', color='tab:blue', label=f'Feeder from Bus {bus[1]}, with control, 20h')

plt.xlabel('Distance from Source Bus (km)', fontsize=16)
plt.ylabel('Voltage Magnitude (p.u.)', fontsize=16)
plt.title(f'Voltage Profile Along Feeder from Bus {bus[1]}', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.show()

print("")

# Plot voltage profile at midday (highest PV, lowest load)
# Tension de tous les bus à un 1/4 d'heure de la journée
# midday = time_steps // 2
# plt.figure(figsize=(12, 6))
# plt.plot(range(res_vm_pu_nc.shape[1]), res_vm_pu_nc.iloc[midday, :], 'ro-', label="No Q=f(U) Control")
# plt.plot(range(res_vm_pu_c.shape[1]), res_vm_pu_c.iloc[midday, :], 'g.-', label="With Q=f(U) Control")
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
# plt.plot(range(res_vm_pu_nc.shape[0]), res_vm_pu_nc.iloc[:, id_bus], 'ro-', label="No Q=f(U) Control")
# plt.plot(range(res_vm_pu_c.shape[0]), res_vm_pu_c.iloc[:, id_bus], 'g.-', label="With Q=f(U) Control")
# plt.axhline(1.1, color='k', linestyle='--', label="Voltage Limits (±10%)")
# plt.axhline(0.9, color='k', linestyle='--')
# plt.title("Voltage Profile at Midday (High PV, Low Load)")
# plt.xlabel("Bus Index")
# plt.ylabel("Voltage [p.u.]")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


# solar_profile, load_profile = create_load_prod_profils(net_no_control)
#
# # Plot le Profile d'ensoleillement
# # Create DataFrame
# df_solar = pd.DataFrame({
#     'time_step': np.arange(time_steps),
#     'solar_scale': solar_profile.df
# })
# # Plot
# plt.figure(figsize=(12, 5))
# plt.plot(df_solar['time_step']/4, df_solar['solar_scale'], label="Profil d'ensoleillement", color='orange')
# plt.title("Profil d'ensoleillement, par pas de temps de 15 min", fontsize=18)
# plt.xlabel("Heures de la journée", fontsize=16)
# plt.ylabel("Proportion de soleil disponible", fontsize=16)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# # Plot le Profil de consommation
# df = pd.DataFrame({
#     'hour': time_steps/4,
#     'load_profile': load_profile.df
# })
#
# # Plot the profile
# plt.figure(figsize=(12, 5))
# plt.plot(df['hour'], df['load_profile'], label="City Load Profile", color="steelblue")
# plt.title("Profil de consommation par pas de temps de 15 minutes", fontsize=22)
# plt.xlabel("Heure de la journée", fontsize=16)
# plt.ylabel("Consommation normalisée", fontsize=16)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()