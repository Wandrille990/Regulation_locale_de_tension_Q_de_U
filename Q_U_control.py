import pandapower as pp
import pandapower.networks as pn
import pandapower.control as ctrl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load existing network (e.g., MV network with PV)
net = pn.mv_oberrhein()

# 2. Set transformer tap to minimum position
for tid in net.trafo.index:
    net.trafo.loc[tid, "tap_pos"] = net.trafo.loc[tid, "tap_min"]

# 3. Create a high-PV / low-load timeseries
time_steps = 24*4
pv_scaling = np.clip(np.sin(np.linspace(0, np.pi, time_steps)), 0.1, 1.0)  # Peaks at midday
load_scaling = np.clip(0.3 + 0.2 * np.cos(np.linspace(0, np.pi, time_steps)+np.pi), 0.2, 0.5)  # Low at midday

# 4. Apply the timeseries
pv_elements = net.sgen[net.sgen['type'] == 'PV'].index
load_elements = net.load.index

#ds = pd.DataFrame(index=range(time_steps), columns=[])
#for el in pv_elements:
#    ds[f"sgen_p_{el}"] = pv_scaling
#for el in load_elements:
#    ds[f"load_p_{el}"] = load_scaling
#
#ds_defrag = ds.copy()

# 5. Create a Q=f(U) Controller (example linear droop function)
class QVoltageDroopControl(ctrl.basic_controller.Controller):
    def __init__(self, net, sgen_index, vm_upper=1.10, vm_lower=0.90, q_max=0.3, in_service=True, order=1, **kwargs):
        super().__init__(net, in_service=in_service, order=order, **kwargs)
        self.sgen_index = sgen_index
        self.vm_upper = vm_upper
        self.vm_lower = vm_lower
        self.q_max = q_max
        self.applied = False

    def control_step(self, net):
        for sg in self.sgen_index:
            bus = net.sgen.at[sg, "bus"]
            vm = net.res_bus.at[bus, "vm_pu"]

            if vm < self.vm_lower:
                q = self.q_max
            elif vm > self.vm_upper:
                q = -self.q_max
            else:
                slope = -2 * self.q_max / (self.vm_upper - self.vm_lower)
                q = slope * (vm - 1.0)

            net.sgen.at[sg, "q_mvar"] = q

        self.applied = True

    def is_converged(self, net):
        # Always return True to apply control only once per power flow
        return self.applied


# 6. Run time series simulations
def run_timeseries(net, control_enabled=False):
    results = []

    if control_enabled:
        ctrl = QVoltageDroopControl(net, sgen_index=pv_elements)

    for t in range(time_steps):
        for el in pv_elements:
            net.sgen.at[el, "scaling"] = pv_scaling[t]
        for el in load_elements:
            net.load.at[el, "scaling"] = load_scaling[t]

        pp.runpp(net, run_control=control_enabled)
        if control_enabled:
            ctrl.applied = False
        results.append(net.res_bus.vm_pu.values.copy())
    return np.array(results)


# 7. Simulate both cases
net_no_control = pn.mv_oberrhein()
net_control = pn.mv_oberrhein()
for tid in net_control.trafo.index:
    net_control.trafo.loc[tid, "tap_pos"] = net_control.trafo.loc[tid, "tap_min"]
for tid in net_no_control.trafo.index:
    net_no_control.trafo.loc[tid, "tap_pos"] = net_no_control.trafo.loc[tid, "tap_min"]

results_nc = run_timeseries(net_no_control, control_enabled=False)
results_c = run_timeseries(net_control, control_enabled=True)

# 8. Plot voltage profile at midday (highest PV, lowest load)
midday = time_steps // 2
plt.figure(figsize=(12, 6))
plt.plot(results_nc[midday], 'r.-', label="No Q=f(U) Control")
plt.plot(results_c[midday], 'g.-', label="With Q=f(U) Control")
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