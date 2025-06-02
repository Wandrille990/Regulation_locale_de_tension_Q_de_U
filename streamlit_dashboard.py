import pandapower.networks as pn
import pandapower.topology as top

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
from collections import namedtuple


def get_voltage_profil(net, res_vm_pu, time=47):

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


net_0 = pn.mv_oberrhein(scenario="generation")

base_res_path = r"C:\Users\Wandrille\Documents\ESIEE\E4FT\Cours E4FT\Multiphysique\Regulation_locale_de_tension_Q_de_U\Resultats"

res_vm_pu_nc = pd.read_csv(base_res_path + r"\res_no_ctrl\res_bus\vm_pu.csv", sep=';').drop("Unnamed: 0", axis=1)
res_vm_pu_c = pd.read_csv(base_res_path + r"\res_ctrl\res_bus\vm_pu.csv", sep=';').drop("Unnamed: 0", axis=1)

_, voltage_profils_nc = get_voltage_profil(net_0, res_vm_pu_nc)
_, voltage_profils_c = get_voltage_profil(net_0, res_vm_pu_c)


st.title("Profil de Tension en fonction du temps")

selected_time = round(st.slider("Choisissez un horaire :", min_value=0., max_value=23.75, value=12., step=0.25)*4)

trace1 = go.Scatter(
    x=voltage_profils_c[318].distances.values,
    y=voltage_profils_c[318].all_voltages.loc[selected_time].values,
    name='Profil de Tension du Bus 318, avec régulation Q=f(U)',
    mode='markers',
    marker=dict(color='rgb(34,163,192)', symbol='circle', size=8)
)

trace2 = go.Scatter(
    x=voltage_profils_nc[318].distances.values,
    y=voltage_profils_nc[318].all_voltages.loc[selected_time].values,
    name='Profil de Tension du Bus 318, sans régulation Q=f(U)',
    mode='markers',
    marker=dict(color='rgb(192,84,34)', symbol='circle')
)

fig = make_subplots()
fig.add_trace(trace1)
fig.add_trace(trace2)
fig.update_layout(xaxis_title="Distance du bus au départ du poste source (km)", yaxis_title="Amplitude de Tension (pu)", yaxis_range=[1.06, 1.115])
fig.update_layout(legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'))
fig['layout'].update(height=600, width=800)

st.plotly_chart(fig)
print("")