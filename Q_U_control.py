import pandapower as pp
import pandapower.networks as pn
import pandapower.control as ctrl
from pandapower.control import ConstControl
from pandapower.control.controller.DERController.der_control import DERController
import pandapower.control.controller.DERController as DERModels
from pandapower.timeseries import run_timeseries, OutputWriter
from pandapower.timeseries.data_sources.frame_data import DFData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1. Load existing network (e.g., MV network with PV)
net = pn.mv_oberrhein(scenario="generation")

# # 2. Set transformer tap to minimum position
# for tid in net.trafo.index:
#     net.trafo.loc[tid, "tap_pos"] = net.trafo.loc[tid, "tap_min"]

# 3. Create a high-PV / low-load timeseries
time_steps = 24*4
pv_scaling = np.clip(np.sin(np.linspace(0, np.pi, time_steps)), 0.1, 1.0)  # Peaks at midday
pv_scaling_matrix = np.tile(pv_scaling.reshape(-1, 1), (1, len(net.sgen.index)))
df_pgen = net.sgen.p_mw.values * pd.DataFrame(pv_scaling_matrix, index=list(range(time_steps)), columns=net.sgen.index)
ds_data_pgen = DFData(df_pgen)

df_pgen_2 = net.sgen.p_mw.values * 2 * pd.DataFrame(pv_scaling_matrix, index=list(range(time_steps)), columns=net.sgen.index)
ds_data_pgen_2 = DFData(df_pgen_2)

load_scaling = np.clip(0.3 + 0.2 * np.cos(np.linspace(0, np.pi, time_steps)+np.pi), 0.2, 0.5)  # Low at midday
load_scaling_matrix = np.tile(load_scaling.reshape(-1, 1), (1, len(net.load.index)))
df_load = net.load.p_mw.values * pd.DataFrame(load_scaling_matrix, index=list(range(time_steps)), columns=net.load.index)
ds_data_load = DFData(df_load)


# 5. Run time series simulations
def opt_run_timeseries(net, ds_gen, ds_load, output_path, control_enabled=False):
    # initialising the outputwriter to save data to excel files in the current folder. You can change this to .json, .csv, or .pickle as well
    ow = OutputWriter(net, output_path=output_path, output_file_type=".csv")
    # adding vm_pu of all buses and line_loading in percent of all lines as outputs to be stored
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')

    const_load_ctrl = ConstControl(net, element_index=net.load.index, element='load', data_source=ds_load, variable='p_mw',
                                   profile_name=ds_load.df.columns, level=1, order=1)
    const_sgen_ctrl = ConstControl(net, element_index=net.sgen.index, element='sgen', data_source=ds_gen, variable='p_mw',
                                   profile_name=ds_gen.df.columns, level=1, order=2)

    if control_enabled:
        der_ctrl = DERController(net, element_index=net.sgen.index, element='sgen', data_source=ds_gen, profile_name=ds_gen.df.columns,
                                 q_model=DERModels.QModelConstQ(-0.33), level=2)  # pqv_area=DERModels.PQVArea4120V2()
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

results_nc = pd.read_csv(r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats\res_no_ctrl\res_bus\vm_pu.csv", sep=';')
results_nc = results_nc.drop("Unnamed: 0", axis=1)
results_c = pd.read_csv(r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats\res_ctrl\res_bus\vm_pu.csv", sep=';')
results_c = results_c.drop("Unnamed: 0", axis=1)

# 7. Plot voltage profile at midday (highest PV, lowest load)
midday = time_steps // 2
plt.figure(figsize=(12, 6))
plt.plot(range(results_nc.shape[1]), results_nc.iloc[midday, :], 'ro-', label="No Q=f(U) Control")
plt.plot(range(results_c.shape[1]), results_c.iloc[midday, :], 'g.-', label="With Q=f(U) Control")
plt.axhline(1.1, color='k', linestyle='--', label="Voltage Limits (±10%)")
plt.axhline(0.9, color='k', linestyle='--')
plt.title("Voltage Profile at Midday (High PV, Low Load)")
plt.xlabel("Bus Index")
plt.ylabel("Voltage [p.u.]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("")