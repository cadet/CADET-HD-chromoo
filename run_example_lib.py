#!/usr/bin/env python3
"""
Chromoo example using the library API directly — no YAML config file needed.

Assumes chromoo is installed (e.g. `pip install -e .` from the repo root).

Run from anywhere:
    python run_example_lib.py

Outputs are written next to this script under results/
"""

import shutil
import numpy as np
from pathlib import Path
from datetime import datetime as dt

from pymoo.util.termination.default import MultiObjectiveDefaultTermination

import chromoo.post as post
from chromoo import ChromooProblem, AlgorithmFactory, ChromooCallback
from chromoo.cache import Cache
from chromoo.log import Logger
from chromoo.parameter import Parameter
from chromoo.objective import Objective
from chromoo.cadetSimulation import CadetSimulation
from chromoo.simulation import load_file
from chromoo.transforms import transform_array
from addict import Dict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
EXAMPLE_DIR  = SCRIPT_DIR / "examples" / "10k-mono-1d-p2"
SIM_FILE     = EXAMPLE_DIR / "10k-mono.mono1d.yaml"
CHROMA_FILE  = EXAMPLE_DIR / "chromatogram-corrected.csv"
WORK_DIR     = SCRIPT_DIR / "results" / "10k-mono-1d-p2"
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
# Change to EXAMPLE_DIR only while constructing Objective (it reads CHROMA_FILE
# relative to cwd). Everything else uses absolute paths.
import os
os.chdir(EXAMPLE_DIR)

parameters = [
    Parameter(
        name      = "axial_dispersion",
        path      = "input.model.unit_002.col_dispersion",
        min_value = 1.0e-9,
        max_value = 1.0e-4,
        length    = 1,
    )
]

objectives = [
    Objective(
        name     = "outlet",
        filename = str(CHROMA_FILE),
        path     = "output.solution.unit_003.solution_outlet_comp_000",
        score    = "sse",
    )
]

os.chdir(WORK_DIR)

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
sim_data = load_file(str(SIM_FILE))
simulation = CadetSimulation()
simulation.root = sim_data.root

# Align solver time grid to the reference chromatogram times
t0 = objectives[0].x0
simulation.root.input.solver.sections.section_times = [float(t0.min()), float(t0.max())]
simulation.root.input.solver.user_solution_times = t0

assert all(obj.verify(simulation) for obj in objectives)

# Use all cores for the pool, 1 thread per CADET instance
nproc = 2
simulation.root.input.solver.nthreads = 1

# ---------------------------------------------------------------------------
# Optimization settings
# ---------------------------------------------------------------------------
PARAMETER_TRANSFORM = "lognorm"
NPROC               = 2
TEMP_DIR            = Path("/dev/shm/chromoo")
STORE_TEMP          = False

from addict import Dict as ADict
algorithm_config = ADict()
algorithm_config.name        = "unsga3"
algorithm_config.pop_size    = 10
algorithm_config.n_offsprings = 10
algorithm_config.n_obj       = sum(o.n_obj for o in objectives)
algorithm_config.init_sobol  = False

termination = MultiObjectiveDefaultTermination(
    x_tol       = 1e-12,
    cv_tol      = 1e-6,
    f_tol       = 1e-9,
    nth_gen     = 2,
    n_last      = 5,
    n_max_gen   = 20,
    n_max_evals = 10000,
)

# ---------------------------------------------------------------------------
# Build problem and cache (cache needs a config-like object with certain attrs)
# ---------------------------------------------------------------------------
class _Config:
    """Minimal config-like namespace that Cache expects."""
    def __init__(self):
        self.parameters        = parameters
        self.objectives        = objectives
        self.parameter_names   = [n for p in parameters for n in p.names]
        self.objective_names   = [n for o in objectives for n in o.names]
        self.n_par             = sum(p.length for p in parameters)
        self.n_obj             = sum(o.n_obj for o in objectives)
        self.par_min_values    = [v for p in parameters for v in p.min_value]
        self.par_max_values    = [v for p in parameters for v in p.max_value]
        self.parameter_transform = PARAMETER_TRANSFORM
        self.simulation        = simulation

config = _Config()
cache  = Cache(config)

prob = ChromooProblem(
    simulation,
    parameters,
    objectives,
    nproc      = NPROC,
    tempdir    = str(TEMP_DIR),
    store_temp = STORE_TEMP,
    transform  = PARAMETER_TRANSFORM,
)

# ---------------------------------------------------------------------------
# Run (with checkpoint resume support)
# ---------------------------------------------------------------------------
logger = Logger()
logger.info(f"Starting chromoo at {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Working directory: {Path.cwd()}")

checkpoint_file = Path("checkpoint.npy")
if checkpoint_file.is_file():
    logger.info(f"Resuming from checkpoint: {checkpoint_file}")
    algo, = np.load(str(checkpoint_file), allow_pickle=True).flatten()
    algo.problem.nproc = NPROC
else:
    logger.info("Starting optimization from scratch.")
    algo = AlgorithmFactory(algorithm_config).get_algorithm()
    algo.setup(prob, termination, callback=ChromooCallback(cache), seed=1, verbose=True)

while algo.has_next():
    algo.next()
    np.save("checkpoint", algo)

res = algo.result()

fitted = transform_array(
    res.X, prob.min_values, prob.max_values, PARAMETER_TRANSFORM, mode="inverse"
)
logger.info(f"Took {res.exec_time:.2f} s to terminate after {len(cache.opt_Fs)} generations.")
logger.info(f"Fitted parameters: {fitted}")
logger.info(f"Objective scores (SSE): {res.F}")

if not STORE_TEMP:
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------
postdir = Path("post")
postdir.mkdir(exist_ok=True)

opts = post.load_dataframe_sort(
    "opts.csv", config.objective_names,
    sort_by="rms",
    rename_columns=config.parameter_names + config.objective_names,
)
pops = post.load_dataframe_sort(
    "populations", config.objective_names,
    sort_by=None,
    rename_columns=["generation"] + config.parameter_names + config.objective_names,
)

opts.to_csv(postdir / "opts_rms.csv")

post.convergence(pops, "rms", postdir)
for par in config.parameter_names:
    post.convergence(pops, par, postdir=postdir, name=f"convergence_{par}")
for obj in config.objective_names:
    post.convergence(pops, obj, postdir=postdir, name=f"convergence_{obj}")

post.violin(opts[opts.columns[: config.n_par]], postdir=postdir, percentile=100)

sims_opts = post.run_sims_parallel(opts, config, NPROC, postdir=postdir, suffix="opts")
post.performance_range_split(sims_opts, config, postdir=postdir)
post.performance_split([sims_opts[0]], config, postdir=postdir)

post.line_plot(opts[config.parameter_names[:10]], f"{postdir}/line.pdf", marker="o", ls="dashed")

logger.info(f"Postprocessing plots written to {postdir}/")
logger.info("Done.")
