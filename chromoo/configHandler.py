"""

ConfigHandler class.

contract:
    - must read and store values from yaml config
    - must provide easy access to deep/nested values (deep_get -> get)

"""

from ruamel.yaml import YAML
from functools import reduce

from pathlib import Path

from chromoo.log import Logger
from chromoo.utils import readArray, readChromatogram
from chromoo.simulation import load_file

from addict import Dict
from chromoo.parameter import Parameter
from chromoo.objective import Objective
from chromoo.cadetSimulation import CadetSimulation

from typing import Any

from itertools import chain
import numpy as np

class ConfigHandler:

    def __init__(self):
        self.config = {}
        self.yaml=YAML(typ='safe')
        self.logger = Logger()
        self.logger.info("Creating config.")

    def read(self, fname):
        """
            Read the config yaml file, save into config dict
        """
        self.config = Dict(self.yaml.load(Path(fname)))

    def get(self, keys, default=None, vartype=None, choices=[], wrapper=None) -> Any:
        """
        Simpler syntax to get deep values from a dictionary
        > config.get('key1.key2.key3', defaultValue)

        - typechecking
        - value restriction
        """
        value = reduce(lambda d, key: d.get(key, None) if isinstance(d, dict) else None, keys.split("."), self.config)

        if value is None:
            if default != None:
                self.logger.warn(keys, 'not specified! Defaulting to', str(default) or 'None (empty string)')
                value = default

        if vartype:
            if not isinstance(value, vartype):
                self.logger.die(keys, 'has invalid type!', str(type(value)), 'instead of', str(vartype))

        if choices:
            if value not in choices:
                self.logger.die(keys, 'has invalid value! Must be one of ', str(choices))

        if wrapper:
            value = wrapper(value)

        return value

    def load(self):
        """
        Assign values from the config dict to the object's attributes
        Centralizes the config value and type checking
        """
        self.filename   =  self.get('filename', vartype=str())
        self.nproc      =  self.get('nproc', vartype=int, default =4)
        self.store_temp =  self.get('store_temp', vartype=bool, default=False)
        self.temp_dir =  self.get('temp_dir', vartype=str(), default='/dev/shm/chromoo')

        self.load_checkpoint = self.get('load_checkpoint', vartype=str, default='.', wrapper=Path)
        self.force_checkpoint_continue = self.get('force_checkpoint_continue', vartype=bool, default=False)

        self.parameters = []
        for param in self.get('parameters', vartype=list) or []:
            self.parameters.append(Parameter(**param))

        self.par_min_values = []
        self.par_max_values = []

        # chaining min and max values
        for p in self.parameters:
            self.par_min_values = list(chain(self.par_min_values, p.min_value))
            self.par_max_values = list(chain(self.par_max_values, p.max_value))

        self.objectives = []
        for param in self.get('objectives', vartype=list) or []:
            self.objectives.append(Objective(**param))

        self.parameter_names = []
        self.objective_names = []

        for p in self.parameters:
            self.parameter_names.extend(p.names)

        for o in self.objectives:
            self.objective_names.extend(o.names)


        self.n_par = sum( p.length for p in self.parameters )
        self.n_obj = sum( o.n_obj for o in self.objectives )


        # If i'm passing a dict to a class, might be better to take the full
        # dict, and then adjust the important subfields. Otherwise, I can just manually
        # constrain the subfields for better sanity.
        # self.algorithm = Dict(self.get('algorithm', None, dict()))
        self.algorithm = Dict()

        self.algorithm.name = self.get('algorithm.name', 'unsga3', str(), ['unsga3', 'nsga3'])
        self.algorithm.pop_size= self.get('algorithm.pop_size', 10, vartype=int)
        self.algorithm.n_offsprings = self.get('algorithm.n_offsprings', self.algorithm.pop_size, vartype=int)
        self.algorithm.n_obj = self.n_obj
        self.algorithm.init_sobol = self.get('algorithm.init_sobol', False, bool)

        self.parameter_transform = self.get('transforms.parameters', 'none', str, ['none', 'lognorm', 'norm'])
        # self.objective_transform = self.get('transforms.objectives', 'none', str, ['none', 'mean', 'geometric'])

        self.termination = Dict()
        self.termination.x_tol = self.get('termination.x_tol', 1e-12, float)
        self.termination.cv_tol = self.get('termination.cv_tol', 1e-6, float)
        self.termination.f_tol = self.get('termination.f_tol', 1e-10, float)
        self.termination.nth_gen = self.get('termination.nth_gen', 2, int)
        self.termination.n_last = self.get('termination.n_last', 5, int)
        self.termination.n_max_gen = self.get('termination.n_max_gen', 100, int)
        self.termination.n_max_evals = self.get('termination.n_max_evals', 1000, int)

    def construct_simulation(self):
        self.simulation = CadetSimulation()
        self.simulation.root = load_file(self.filename).root

        t0 = self.objectives[0].x0
        assert all(map(lambda obj: np.allclose(obj.x0, t0), self.objectives))

        self.simulation.root.input.solver.sections.section_times = [min(t0), max(t0)]
        self.simulation.root.input.solver.user_solution_times = t0

        assert all( [ obj.verify(self.simulation) for obj in self.objectives ] )

        if self.nproc > 1: 
            self.simulation.root.input.solver.nthreads = 1
        else: 
            self.simulation.root.input.solver.nthreads = -1

