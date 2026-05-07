#!/usr/bin/env python3
"""
Example script to run chromoo optimization on the 10k-mono-1d-p2 example.

The example fits the axial dispersion coefficient of a 1D chromatography model
against a reference chromatogram using the UNSGA3 multi-objective algorithm.

Run from the chromoo root directory:
    python run_example.py

Results:
    - checkpoint.npy  : saved after every generation, can be resumed
    - Fitted parameters and scores are printed at the end
"""

import os
import sys
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Change to the example directory so that relative paths in chromoo.yaml work
# (simulation file and reference chromatogram are looked up relative to cwd)
# ---------------------------------------------------------------------------
# EXAMPLE_DIR = Path(__file__).parent / "examples" / "10k-mono-1d-p2_DGP4Z8"
EXAMPLE_DIR = Path(__file__).parent / "examples" / "mono-mono-clipped-dax5-dr5-df"
os.chdir(EXAMPLE_DIR)

# # ---------------------------------------------------------------------------
# # Inline config (equivalent to a chromoo.yaml in this directory)
# # ---------------------------------------------------------------------------
# CONFIG_YAML = """\
# filename: chromoo.yaml
# nproc: 2
# store_temp: false
# transforms:
#   parameters: lognorm
# parameters:
#     - name: axial_dispersion
#       length: 1
#       path: input.model.unit_002.col_dispersion
#       min_value: 1.0e-9
#       max_value: 1.0e-4
# objectives:
#     - name: outlet
#       filename: chromatogram-corrected.csv
#       score: sse
#       path: output.solution.unit_003.solution_outlet_comp_000
# algorithm:
#   name: unsga3
#   pop_size: 10
# termination:
#   f_tol: 1e-9
#   nth_gen: 2
#   n_last: 5
#   n_max_gen: 20
#   n_max_evals: 10000
# """

# # Write config to a temporary file in the example directory
# config_path = Path("chromoo_example_run.yaml")
# config_path.write_text(CONFIG_YAML)

config_path = Path("chromoo.yaml")

# ---------------------------------------------------------------------------
# Run the optimization (mirrors the logic in bin/chromoo)
# ---------------------------------------------------------------------------
from datetime import datetime as dt
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from SALib.sample import sobol_sequence

from chromoo import ChromooProblem, AlgorithmFactory, ConfigHandler, ChromooCallback
from chromoo.cache import Cache
from chromoo.log import Logger
from chromoo.transforms import transform_array

logger = Logger()
logger.info(f"Starting chromoo example at {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Working directory: {Path.cwd()}")

config = ConfigHandler()
config.read(str(config_path))
config.load()
config.construct_simulation()

cache = Cache(config)

prob = ChromooProblem(
    config.simulation,
    config.parameters,
    config.objectives,
    nproc=config.nproc,
    tempdir=config.temp_dir,
    store_temp=config.store_temp,
    transform=config.parameter_transform,
)

term = MultiObjectiveDefaultTermination(
    x_tol=config.termination.x_tol,
    cv_tol=config.termination.cv_tol,
    f_tol=config.termination.f_tol,
    nth_gen=config.termination.nth_gen,
    n_last=config.termination.n_last,
    n_max_gen=config.termination.n_max_gen,
    n_max_evals=config.termination.n_max_evals,
)

checkpoint_file = Path("checkpoint.npy")
if checkpoint_file.is_file():
    logger.info(f"Resuming from checkpoint: {checkpoint_file}")
    algo, = np.load(str(checkpoint_file), allow_pickle=True).flatten()
    algo.problem.nproc = config.nproc
else:
    logger.info("Starting optimization from scratch.")
    algo = AlgorithmFactory(config.algorithm).get_algorithm()
    algo.setup(prob, term, callback=ChromooCallback(cache), seed=1, verbose=True)

while algo.has_next():
    algo.next()
    np.save("checkpoint", algo)

res = algo.result()

logger.info(
    f"Took {res.exec_time:.2f} s ({res.exec_time / 3600:.4f} h) "
    f"to terminate after {len(cache.opt_Fs)} generations."
)
logger.info(
    f"Fitted parameters: "
    f"{transform_array(res.X, prob.min_values, prob.max_values, config.parameter_transform, mode='inverse')}"
)
logger.info(f"Objective scores (SSE): {res.F}")

if not config.store_temp:
    import shutil
    shutil.rmtree(prob.tempdir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------
import chromoo.post as post

postdir = Path("post")
postdir.mkdir(exist_ok=True)

opts = post.load_dataframe_sort("opts.csv", config.objective_names, sort_by="rms", rename_columns=config.parameter_names + config.objective_names)
pops = post.load_dataframe_sort("populations", config.objective_names, None, ["generation"] + config.parameter_names + config.objective_names)

opts.to_csv(postdir / "opts_rms.csv")

post.convergence(pops, "rms", postdir)
for par in config.parameter_names:
    post.convergence(pops, par, postdir=postdir, name=f"convergence_{par}")
for obj in config.objective_names:
    post.convergence(pops, obj, postdir=postdir, name=f"convergence_{obj}")

post.violin(opts[opts.columns[0:config.n_par]], postdir=postdir, percentile=100)

# sims_opts = post.run_sims_parallel(opts, config, config.nproc, postdir=postdir, suffix="opts")
sims_opts = post.run_sims(opts, config, postdir=postdir, suffix="opts")
post.performance_range_split(sims_opts, config, postdir=postdir)
post.performance_split([sims_opts[0]], config, postdir=postdir)

post.line_plot(opts[config.parameter_names[:10]], f"{postdir}/line.pdf", marker="o", ls="dashed")

logger.info(f"Postprocessing plots written to {postdir}/")

config_path.unlink(missing_ok=True)
logger.info("Done.")
