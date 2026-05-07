from chromoo.cadetSimulation import CadetSimulation
import numpy as np
from joblib import Memory

from matplotlib import pyplot as plt
import matplotlib as mpl
from cycler import cycler

CACHE_DIR = f"cache"
memory = Memory(location=CACHE_DIR, verbose=0)

## Caching helps us iterate on post-proc faster. Delete CACHE_DIR directory for reset.
@memory.cache
def sim_load_run_post():
    sim = CadetSimulation()
    sim.load_file('./long.poly2d.yaml')
    sim.save()
    sim.run_simulation()
    sim.load()
    return sim

sim = sim_load_run_post()

unit_id = 1
unit_str = f'unit_{unit_id:03d}'

sim.post_mass_solid_all_partypes(unit_id)
data = sim.root.output.post[f'unit_001'].post_mass_solid_all_partypes
times = sim.root.input.solver.user_solution_times

data_summed = np.sum(data, axis=1)
fig, ax = plt.subplots()
cmap = mpl.cm.get_cmap(name='tab20')
color_cycler = cycler(color=cmap.colors)
ax.set_prop_cycle(color_cycler)

for i in range(5):
    t,c = np.loadtxt(f'./reference/corrected_0.9441/ref{i}_yscaled0.25.csv', delimiter=',').T
    ax.plot(t,c, label=f"rad_{i}")
    ax.plot(times, data_summed[:,i], ls='dashed')

plt.legend()
plt.savefig('plot.pdf')
