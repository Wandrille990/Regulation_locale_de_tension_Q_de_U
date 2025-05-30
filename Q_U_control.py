import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top
from pandapower.control import ConstControl
import pandapower.control.controller.DERController as DERModels
from pandapower.timeseries import run_timeseries, OutputWriter
from pandapower.timeseries.data_sources.frame_data import DFData

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1. Load existing network (e.g., MV network with PV)
net_0 = pn.mv_oberrhein(scenario="generation")

# # 2. Set transformer tap to minimum position
# for tid in net.trafo.index:
#     net.trafo.loc[tid, "tap_pos"] = net.trafo.loc[tid, "tap_min"]

# 3. Create a high-PV / low-load timeseries
time_steps = 24*4
pv_scaling = np.clip(np.sin(np.linspace(0, np.pi, time_steps)), 0.1, 1.0)  # Peaks at midday
pv_scaling_matrix = np.tile(pv_scaling.reshape(-1, 1), (1, len(net_0.sgen.index)))
df_pgen = net_0.sgen.p_mw.values * pd.DataFrame(pv_scaling_matrix, index=list(range(time_steps)), columns=net_0.sgen.index)
ds_data_pgen = DFData(df_pgen)

df_pgen_2 = net_0.sgen.p_mw.values * 1 * pd.DataFrame(pv_scaling_matrix, index=list(range(time_steps)), columns=net_0.sgen.index)
ds_data_pgen_2 = DFData(df_pgen_2)

load_scaling = np.clip(0.3 + 0.2 * np.cos(np.linspace(0, np.pi, time_steps)+np.pi), 0.2, 0.5)  # Low at midday
load_scaling_matrix = np.tile(load_scaling.reshape(-1, 1), (1, len(net_0.load.index)))
df_load = net_0.load.p_mw.values * pd.DataFrame(load_scaling_matrix, index=list(range(time_steps)), columns=net_0.load.index)
ds_data_load = DFData(df_load)


# 5. Run time series simulations
def opt_run_timeseries(net, ds_gen, ds_load, output_path, control_enabled=False):
    # initialising the outputwriter to save data to excel files in the current folder. You can change this to .json, .csv, or .pickle as well
    ow = OutputWriter(net, output_path=output_path, output_file_type=".csv")
    # adding vm_pu of all buses and line_loading in percent of all lines as outputs to be stored
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_sgen', 'p_mw')
    ow.log_variable('res_sgen', 'q_mvar')

    const_pload_ctrl = ConstControl(net, element_index=net.load.index, element='load', data_source=ds_load, variable='p_mw',
                                    profile_name=ds_load.df.columns, level=1, order=1)
    const_psgen_ctrl = ConstControl(net, element_index=net.sgen.index, element='sgen', data_source=ds_gen, variable='p_mw',
                                    profile_name=ds_gen.df.columns, level=1, order=2)
    const_ssgen_ctrl = ConstControl(net, element_index=net.sgen.index, element='sgen', data_source=ds_gen, variable='sn_mva',
                                    profile_name=ds_gen.df.columns, level=1, order=3)

    if control_enabled:
        vm_points_pu = np.array([0, 0.96, 0.9725, 1.0375, 1.05, 2])  # DTR Enedis-NOI-RES_60E
        cosphi_points = np.array([0.9285, 0.9285, 0, 0, -0.9285, -0.9285])  # tan(phi) = jusqu'à 0,4 en sous tension, jusqu'à -0,4 en sur tension
        der_ctrl = DERModels.DERController(net, element_index=net.sgen.index, element='sgen', data_source=ds_gen, profile_name=ds_gen.df.columns,
                                           q_model=DERModels.QModelCosphiVCurve(DERModels.CosphiVCurve(vm_points_pu, cosphi_points)),
                                           pqv_area=DERModels.PQVArea4110(), level=2)  #   # QModelCosphiVCurve ou QModelQVCurve
        run_timeseries(net)
    else:
        run_timeseries(net)

# 6. Simulate both cases
output_path_no_ctrl = r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats\res_no_ctrl"
output_path_ctrl = r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats\res_ctrl"

net_no_control = pn.mv_oberrhein(scenario="generation")
net_control = pn.mv_oberrhein(scenario="generation")

for trsf_id in net_control.trafo.index:
    net_control.trafo.loc[trsf_id, "tap_pos"] = -5  # net_control.trafo.loc[trsf_id, "tap_min"]
for trsf_id in net_no_control.trafo.index:
    net_no_control.trafo.loc[trsf_id, "tap_pos"] = -5  # net_no_control.trafo.loc[trsf_id, "tap_min"]

opt_run_timeseries(net_no_control, ds_data_pgen_2, ds_data_load, output_path_no_ctrl, control_enabled=False)
opt_run_timeseries(net_control, ds_data_pgen_2, ds_data_load, output_path_ctrl, control_enabled=True)

res_vm_pu_nc = pd.read_csv(r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats\res_no_ctrl\res_bus\vm_pu.csv", sep=';')
res_vm_pu_nc = res_vm_pu_nc.drop("Unnamed: 0", axis=1)
res_vm_pu_c = pd.read_csv(r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats\res_ctrl\res_bus\vm_pu.csv", sep=';')
res_vm_pu_c = res_vm_pu_c.drop("Unnamed: 0", axis=1)

res_p_mw_nc = pd.read_csv(r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats\res_no_ctrl\res_sgen\p_mw.csv", sep=';')
res_p_mw_nc = res_p_mw_nc.drop("Unnamed: 0", axis=1)
res_p_mw_c = pd.read_csv(r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats\res_ctrl\res_sgen\p_mw.csv", sep=';')
res_p_mw_c = res_p_mw_c.drop("Unnamed: 0", axis=1)

print(f"Energie totale produite sur la journée sans contrôle : {round(res_p_mw_nc.sum().sum()*15/60, 1)} MWh")
print(f"Energie totale produite sur la journée avec contrôle : {round(res_p_mw_c.sum().sum()*15/60, 1)} MWh")

def get_voltage_profil(net, res, time):

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

    for source_bus, buses in supplied_buses.items():
        # Calculate distances from the source bus
        distances = top.calc_distance_to_bus(net, source_bus).loc[buses]

        # Extract voltage magnitudes for the buses
        voltages = res.loc[time, [str(bus) for bus in buses]]

        # Sort distances and corresponding voltages for plotting
        sorted_indices = distances.sort_values().index
        sorted_distances = distances.loc[sorted_indices]
        sorted_voltages = voltages.loc[[str(id) for id in sorted_indices]]
        sorted_dist_volt[source_bus] = [sorted_distances, sorted_voltages]

    return sorted_dist_volt

midday = time_steps // 2
voltage_profils_nc = get_voltage_profil(net_no_control, res_vm_pu_nc, midday)
voltage_profils_c = get_voltage_profil(net_control, res_vm_pu_c, midday)

plt.figure(figsize=(10, 6))
bus = list(voltage_profils_c.keys())
plt.scatter(voltage_profils_c[bus[1]][0], voltage_profils_c[bus[1]][1], marker='o', label=f'Feeder from Bus {bus[1]}, with control')
plt.scatter(voltage_profils_nc[bus[1]][0], voltage_profils_nc[bus[1]][1], marker='o', label=f'Feeder from Bus {bus[1]}, without control')
plt.xlabel('Distance from Source Bus (km)')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.title(f'Voltage Profile Along Feeder from Bus {bus}')
plt.grid(True)
plt.legend()
plt.show()

# 7. Plot voltage profile at midday (highest PV, lowest load)
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