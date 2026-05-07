# chromoo

[![DOI](https://zenodo.org/badge/611842983.svg)](https://doi.org/10.5281/zenodo.14876412)

`chromoo` is a chromatography multi-objective optimization tool built on `Cadet-Core` and `pymoo==0.5`.

# Installation

```
# Install cadet. This can be done via conda as below 
# or directly from source https://github.com/modsim/CADET
conda config --add channels conda-forge
conda config --set channel_priority strict

# Install python dependencies
pip install -r requirements.txt

# Install this package. Use -e for an editable install.
pip install [-e] . 
```

# Usage

Chromoo requires a YAML config file of the following form.

```yaml
filename: 10k-mono.mono1d.h5
load_checkpoint: checkpoint.npy
force_checkpoint_continue: false
nproc: 4
store_temp: false
transforms: 
  parameters: lognorm
  objectives: geometric
parameters:
    - name: axial
      length: 1
      path: input.model.unit_002.col_dispersion
      min_value: 1.0e-9
      max_value: 1.0e-4
objectives: 
    - name: outlet
      filename: chromatogram-from-xns.csv
      # times: timesteps.txt
      score: sse
      path: output.solution.unit_003.solution_outlet_comp_000
algorithm: 
  name: nsga3
  pop_size: 10
termination:
  x_tol: 1e-8
  cv_tol: 1e-6
  f_tol: 1e-9
  nth_gen: 2
  n_last: 10
  n_max_gen: 10
  n_max_evals: 100000
```

# Notes
- It runs multiple cadet simulations from a pool size of `nproc` for every evaluation of a population.
- parameters and objectives are **lists**
- Objective targets can be provided as an (times,values) csv file in `objectives.filename` or with the times separately specified in `objectives.times`
    - chromatograms already contain times, so it's easier to just provide the filename
- The `solution_times` section of the provided cadet simulation will be changed to match those of `objectives[0]` exactly.
- Recommended population sizes for n-dimensional problems is 100*n
- Don't fit porosity and velocity together. You can fit porosity and flowrate instead
- Provided examples, while valid, are *NOT* guaranteed to be correct as the software is not guaranteed to be stable in terms of development and backwards compatibility.
- Checkpoints are saved at every generation by default.
- Use `force_checkpoint_continue` to force the algorithm to continue from a _terminated_ checkpoint. Helpful if you made the termination criteria stricter than required.
- Be careful when resuming from a checkpoint. Any changes to problem parameters might not be reflected because the algorithm/problem is fully restored from the checkpoint

# Known Issues
- Reading inputs from YAML loads strings as `str` and from h5 files we get `numpy.bytes_`. CADET-Python run_load() uses load_results() instead of full load(). So if we check for input string values after simulation, the type of it depends on whether we use full load() or load_results() since we deal with YAML files as well. So we have to consider whether we deal with strings or bytestrings. Simple solution: Don't use run_load in scripts.
- Loading checkpoints also loads the previous values for all/most parameters. So if nproc is updated before loading, the new value isn't used.
