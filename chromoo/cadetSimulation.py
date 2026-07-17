from addict import Dict
from cadet import Cadet
from chromoo.utils import deep_get, keystring_todict, readChromatogram, readArray, pairwise
from pathlib import Path
from ruamel.yaml import YAML
from typing import Any
from typing import TypeVar, Callable, Any, Optional

from chromoo.scores import scores_dict

import multiprocessing as mp

import re
import os
import random
import string
import numpy as np
import subprocess

T = TypeVar("T")

cadetpath = subprocess.run(['which', 'cadet-cli'], capture_output=True, text=True ).stdout.strip()
Cadet.cadet_path = cadetpath

class CadetSimulation(Cadet): 
    def get(self, path:str, vartype:Callable[[Any],T]=Any, default=None, choices=[]) -> T: 
        """ Returns the value of a dot-separated path in the simulation data """
        return deep_get(self.root, path, vartype=vartype, default=default, choices=choices)

    def set(self, path:str, value:Any):
        """ Set a value in the simulation data at a given dot-separated path """
        dic = self.root

        for key in path.split('.'):
            dic = dic[key]

        dic = value

    def take(self, path:str, indices, axis:int): 
        """ Take data at indices on axis of an n-dimensional array """
        array = self.get(path, vartype=np.array)
        return np.take(array, indices=indices, axis=axis)

    def take_average(self, path:str, take_indices, take_axis:int, avg_axis:int, avg_x:list):
        """ Take data slice, then average it along another axis """
        array = self.get(path, vartype=np.array)
        taken = np.take(array, indices=take_indices, axis=take_axis)
        return np.trapz(taken, x=avg_x, axis=avg_axis) / (max(avg_x) - min(avg_x)) 

    def get_shape_pre(self, path:str): 
        """ A hack to pre-emptively know what the cadet output path shape is """
        unit = re.findall(r'unit_\d+', path)[0]
        output_type = path.split('.')[-1].split('_')[1]
        nts = len(self.root.input.solver.user_solution_times)

        ncol = self.root.input.model[unit].discretization.ncol
        nrad = self.root.input.model[unit].discretization.nrad
        ncomp = self.root.input.model[unit].ncomp
        npar = self.root.input.model[unit].npar

        ## WARNING: Haven't doublechecked this
        if output_type == 'outlet': 
            return (nts,)
        elif output_type == 'bulk': 
            return tuple(filter(None, (nts, ncol, nrad, ncomp)))
        elif output_type == 'particle': 
            return tuple(filter(None, (nts, ncol, nrad, npar)))
        elif output_type == 'solid': 
            return tuple(filter(None, (nts, ncol, nrad, npar)))
        elif path.split('.')[1] == 'post': 
            return tuple(filter(None, (nts, ncol, nrad)))
        else: 
            raise NotImplementedError


    def load_file(self, fname: str) -> None: 
        """ Load a simulation from either h5 or yaml file """
        ext = Path(fname).suffix

        if ext == ".h5":
            self.load_h5(fname)
        elif ext == ".yaml" or ext == ".yml" :
            self.load_yaml(fname)
        else:
            raise(RuntimeError('Invalid simulation file!'))

    def load_h5(self, filename: str):
        """ Load a cadet simulation file """
        self.filename = filename
        self.load()

    def load_yaml(self, filename:str):
        """ Load a YAML config for CADET """
        yaml=YAML(typ='safe')
        yaml.default_flow_style=False
        ext = Path(filename).suffix

        self.filename = filename.replace(ext, '.h5')
        self.root = Dict(yaml.load(Path(filename)))

    def update_parameters(self, x, parameters): 
        """
        Update parameters with values in x.
        """
        prev_len = 0
        for p in parameters:
            cur_len = p.length

            if p.type == 'element' :
                arr = self.root
                for key in p.path.split('.'):
                    arr = arr[key]
                value = x[prev_len : prev_len + cur_len]
                for idx_len,idx_arr in zip(range(cur_len), p.index):
                    arr[idx_arr] = value[idx_len]
                cur_dict = keystring_todict(p.path, arr)

                ## NOTE: Hack to copy flowrate values within the connections matrix
                if p.copy_to_path: 
                    arr2 = self.root
                    for key in p.copy_to_path.split('.'):
                        arr2 = arr2[key]
                    for idx_len,idx_copy_list in zip(range(cur_len), p.copy_to_index):
                        for idx_copy in idx_copy_list: 
                            arr2[idx_copy] = value[idx_len]
                    cur_dict.update(keystring_todict(p.copy_to_path, arr2))

            else:
                cur_dict = keystring_todict(p.path, x[prev_len : prev_len + cur_len])

                ## NOTE: Don't need this, so commented for safety, but written for generality
                # if p.copy_to_path: 
                #     cur_dict.update(keystring_todict(p.copy_to_path, x[prev_len : prev_len + cur_len]))

            self.root.update(cur_dict)
            prev_len += p.length

    def save_run_load(self, overwrite=True):
        p = Path(self.filename)

        if p.exists(): 
            if not overwrite:
                print("File {} already exists! Use overwrite flag to continue.")
                exit(-1)

        self.save()

        return_information = self.run_simulation()
        if return_information.return_code != 0:
            print(f"{self.filename} failed with return code {return_information.return_code}")
            raise RuntimeError("Simulation Failure")

    def run_with_parameters(self, x, parameters, name:Optional[str]=None, tempdir:Path=Path('temp'), store:bool=False): 
        """ 
        Run the simulation with a given set of parameters. 
            -> x: parameter values
            -> parameter: dict with the following keys
                  - name: str
                  - type: scalar | vector | element
                  - length: <int>
                  - index: <int>
                  - path: str # input.model.unit_002.col_dispersion
                  - min_value: float 
                  - max_value: float
        """

        if name:
            self.filename = tempdir.joinpath(name + '.h5')
        else:
            self.filename = tempdir.joinpath('temp' + ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6)) + '.h5')

        self.update_parameters(x, parameters)

        self.save()

        return_information = self.run_simulation()
        if return_information.return_code != 0:
            print(f"{self.filename} failed with return code {return_information.return_code} and error msg {return_information.error_message}")
            print(f"Parameters: {x}\n")
            raise RuntimeError("Simulation Failure")

        if not store:
            os.remove(self.filename)

    def combine_port_breakthroughs(self, unit:int, comp:int=0, switch:int=0): 

        solution_unit = self.root.output.solution[f'unit_{unit:03d}']

        keys = [ key 
                for key in solution_unit.keys()
                if f'comp_{comp:03d}' in key if 'outlet' in key]

        nrad = self.root.input.model[f'unit_{unit:03d}'].discretization.nrad

        solutions = np.stack([ solution_unit[key] for key in keys ])

        assert(len(keys) == nrad)

        connections = self.root.input.model.connections[f'switch_{switch:03d}'].connections
        assert self.root.input.model.connections.connections_include_ports == 1
        connections = np.reshape(connections, (-1,7))
        my_unit_filter = np.asarray([unit])

        # Input flowrates into this unit
        filtered_connections = connections[np.in1d(connections[:,1], my_unit_filter)]

        # Sort by current unit's ports
        filtered_connections = filtered_connections[filtered_connections[:,3].argsort()]
        # Find indices where ports change
        d = np.diff(filtered_connections[:,3])
        e = np.where(d)[0]
        indices = np.r_[0,e+1]
        # Reduce (add) at/upto the given indices
        # Essentially adds all incoming flowrates, since we only extract flowrates
        flowrates = np.add.reduceat(filtered_connections, indices, axis=0)[:,6]

        assert(len(flowrates) == nrad)

        return np.sum(solutions.T * flowrates, axis=1) / sum(flowrates)

    def post_mass_bulk(self, unit:int): 
        """
        Return array of internal mass (num. moles) calculated as 
        $\sum c \cdot V$, where c -> concentration and V -> interstitial volume.

        Conc. within particles is averaged out
        """
        UNIT = self.root.input.model[f'unit_{unit:03d}']
        UNIT_OUT = self.root.output.solution[f'unit_{unit:03d}']

        col_radius = UNIT.col_radius or np.sqrt(UNIT.cross_section_area / (np.pi))

        sol_bulk     = UNIT_OUT.solution_bulk.squeeze()
        vol_bulk     = self.get_vol_array(unit, 'bulk')
        mass_bulk     = sol_bulk     * vol_bulk[np.newaxis, :]

        # WARNING: Somehow broken in addict v2.4 with chromoo-post. Works in v2.3
        # Potentially relevant: https://github.com/mewwts/addict/issues/136
        self.root.output.post[f'unit_{unit:03d}'].post_mass_bulk = mass_bulk 

    def post_mass_par(self, unit:int): 
        """
        Return array of internal mass (num. moles) calculated as 
        $\sum c \cdot V$, where c -> concentration and V -> particle pore volume.
        """
        UNIT = self.root.input.model[f'unit_{unit:03d}']
        UNIT_OUT = self.root.output.solution[f'unit_{unit:03d}']

        col_radius = UNIT.col_radius or np.sqrt(UNIT.cross_section_area / (np.pi))

        sol_particle = UNIT_OUT.solution_particle.squeeze()
        vol_particle = self.get_vol_array(unit, 'particle')

        if UNIT.discretization.par_disc_type == b'EQUIDISTANT_PAR':
            par_radius = UNIT.par_radius
            assert isinstance(par_radius, np.floating) or isinstance(par_radius, float)
            npar = UNIT.discretization.npar
            nShells = npar + 1 #Including r = 0
            rShells = [ par_radius * (n/npar) for n in range(nShells) ]
        else:
            print(UNIT.discretization.par_disc_type)
            raise NotImplementedError

        ## Reversed because of how it is in CADET
        vol_par_shells = np.array([ (r_out**3 - r_in**3)/par_radius**3 for r_out,r_in in pairwise(reversed(rShells)) ])

        # Integrated results per ncol x nrad cells
        # NOTE: Assumes npar is the last axis of sol_*
        sol_particle = np.tensordot(sol_particle, vol_par_shells, 1 )

        mass_particle = sol_particle * vol_particle[np.newaxis, :]

        # WARNING: Somehow broken in addict v2.4 with chromoo-post. Works in v2.3
        # Potentially relevant: https://github.com/mewwts/addict/issues/136
        self.root.output.post[f'unit_{unit:03d}'].post_mass_par = mass_particle

    def post_mass_solid(self, unit:int): 
        """
        Return array of internal mass (num. moles) calculated as 
        $\sum c \cdot V$, where c -> concentration and V -> particle solid volume.
        """
        UNIT = self.root.input.model[f'unit_{unit:03d}']
        UNIT_OUT = self.root.output.solution[f'unit_{unit:03d}']

        col_radius = UNIT.col_radius or np.sqrt(UNIT.cross_section_area / (np.pi))

        sol_solid = UNIT_OUT.solution_solid.squeeze()
        vol_solid = self.get_vol_array(unit, 'solid')

        if UNIT.discretization.par_disc_type == b'EQUIDISTANT_PAR':
            par_radius = UNIT.par_radius
            assert isinstance(par_radius, np.floating) or isinstance(par_radius, float)
            npar = UNIT.discretization.npar
            nShells = npar + 1 #Including r = 0
            rShells = [ par_radius * (n/npar) for n in range(nShells) ]
        else:
            print(UNIT.discretization.par_disc_type)
            raise NotImplementedError

        ## Reversed because of how it is in CADET
        vol_par_shells = np.array([ (r_out**3 - r_in**3)/par_radius**3 for r_out,r_in in pairwise(reversed(rShells)) ])

        # Integrated results per ncol x nrad cells
        # NOTE: Assumes npar is the last axis of sol_*
        sol_solid = np.tensordot(sol_solid, vol_par_shells, 1 )

        mass_solid = sol_solid * vol_solid[np.newaxis, :]

        # WARNING: Somehow broken in addict v2.4 with chromoo-post. Works in v2.3
        # Potentially relevant: https://github.com/mewwts/addict/issues/136
        self.root.output.post[f'unit_{unit:03d}'].post_mass_solid = mass_solid

    def post_mass_total(self, unit:int): 
        """
        Return array of internal mass (num. moles) calculated as 
        $\sum c \cdot V$, where c -> concentration and V -> volume.

        Conc. within particles is averaged out

        ------------ WARNING: simple code, doesn't consider 2D PSD! --------------
        """
        UNIT = self.root.input.model[f'unit_{unit:03d}']
        UNIT_OUT = self.root.output.solution[f'unit_{unit:03d}']

        col_radius = UNIT.col_radius or np.sqrt(UNIT.cross_section_area / (np.pi))

        sol_bulk     = UNIT_OUT.solution_bulk.squeeze()
        sol_particle = UNIT_OUT.solution_particle.squeeze()
        sol_solid    = UNIT_OUT.solution_solid.squeeze()

        vol_bulk     = self.get_vol_array(unit, 'bulk')
        vol_particle = self.get_vol_array(unit, 'particle')
        vol_solid    = self.get_vol_array(unit, 'solid')
        vol_total    = self.get_vol_array(unit, 'total')

        if UNIT.discretization.par_disc_type == b'EQUIDISTANT_PAR':
            par_radius = UNIT.par_radius
            assert isinstance(par_radius, np.floating) or isinstance(par_radius, float)
            npar = UNIT.discretization.npar
            nShells = npar + 1 #Including r = 0
            rShells = [ par_radius * (n/npar) for n in range(nShells) ]
        else:
            print(UNIT.discretization.par_disc_type)
            raise NotImplementedError

        # # WARNING: Hacky. Unsure where components go.
        # if UNIT.unit_type == b'GENERAL_RATE_MODEL_2D': 
        #     # Assumes (nts, ncol, nrad, npar)
        #     par_axis = 3
        # else: 
        #     # Assumes (nts, ncol, npar)
        #     par_axis = 2

        ## Reversed because of how it is in CADET
        vol_par_shells = np.array([ (r_out**3 - r_in**3)/par_radius**3 for r_out,r_in in pairwise(reversed(rShells)) ])

        # Integrated results per ncol x nrad cells
        # NOTE: Assumes npar is the last axis of sol_*
        sol_particle = np.tensordot(sol_particle, vol_par_shells, 1 )
        sol_solid    = np.tensordot(sol_solid, vol_par_shells, 1 )

        assert np.allclose(vol_bulk + vol_particle + vol_solid, vol_total, rtol=1e-6, atol=0.0)

        mass_bulk     = sol_bulk     * vol_bulk[np.newaxis, :]
        mass_particle = sol_particle * vol_particle[np.newaxis, :]
        mass_solid    = sol_solid    * vol_solid[np.newaxis, :]

        # WARNING: Somehow broken in addict v2.4 with chromoo-post. Works in v2.3
        # Potentially relevant: https://github.com/mewwts/addict/issues/136
        self.root.output.post[f'unit_{unit:03d}'].post_mass_total = mass_bulk + mass_particle + mass_solid
        self.root.output.post[f'unit_{unit:03d}'].post_mass_bulk = mass_bulk 
        self.root.output.post[f'unit_{unit:03d}'].post_mass_par = mass_particle
        self.root.output.post[f'unit_{unit:03d}'].post_mass_solid = mass_solid

    def post_mass_solid_all_partypes(self, unit:int):
        """ For polydisperse cases """
        UNIT = self.root.input.model[f'unit_{unit:03d}']
        UNIT_OUT = self.root.output.solution[f'unit_{unit:03d}']
        keys = list(filter(lambda key: 'solution_solid_partype_' in key.lower() ,UNIT_OUT.keys()))
        
        result = []

        ncol = UNIT.discretization.ncol
        nrad = UNIT.discretization.nrad
        npartype = len(UNIT.par_radius)

        assert(npartype == len(keys))

        col_radius = UNIT.col_radius or np.sqrt(UNIT.cross_section_area / (np.pi))

        for ind,key in enumerate(sorted(keys)):
            sol_solid = UNIT_OUT[key].squeeze()

            if len(UNIT.par_type_volfrac) == npartype:
                vol_solid = self.get_vol_array(unit, 'solid') * UNIT.par_type_volfrac[ind]
            elif len(UNIT.par_type_volfrac) == nrad * npartype:
                par_type_volfrac = np.reshape(UNIT.par_type_volfrac, (nrad,npartype))
                vol_solid = self.get_vol_array(unit, 'solid') * par_type_volfrac[:,ind]
            else:
                raise NotImplementedError

            par_radius = UNIT.par_radius[ind]
            npar = UNIT.discretization.npar[ind]

            if UNIT.discretization.par_disc_type == b'EQUIDISTANT_PAR':
                assert isinstance(par_radius, np.floating) or isinstance(par_radius, float)
                nShells = npar + 1 #Including r = 0
                rShells = [ par_radius * (n/npar) for n in range(nShells) ]
            else:
                print(UNIT.discretization.par_disc_type)
                raise NotImplementedError

            ## Reversed because of how it is in CADET
            vol_par_shells = np.array([ (r_out**3 - r_in**3)/par_radius**3 for r_out,r_in in pairwise(reversed(rShells)) ])

            # Integrated results per ncol x nrad cells
            # NOTE: Assumes npar is the last axis of sol_*
            sol_solid = np.tensordot(sol_solid, vol_par_shells, 1 )

            mass_solid = sol_solid * vol_solid[np.newaxis, :]
            result.append(mass_solid)

        # np.sum(list(map(lambda key: UNIT_OUT[key], keys)), axis=0)
        self.root.output.post[f'unit_{unit:03d}'].post_mass_solid_all_partypes = np.sum(result, axis=0)

    def post_mass_par_all_partypes(self, unit:int):
        """ For polydisperse cases """
        UNIT = self.root.input.model[f'unit_{unit:03d}']
        UNIT_OUT = self.root.output.solution[f'unit_{unit:03d}']
        keys = list(filter(lambda key: 'solution_particle_partype_' in key ,UNIT_OUT.keys()))

        result = []

        ncol = UNIT.discretization.ncol
        nrad = UNIT.discretization.nrad
        npartype = len(UNIT.par_radius)

        col_radius = UNIT.col_radius or np.sqrt(UNIT.cross_section_area / (np.pi))

        for ind,key in enumerate(sorted(keys)):

            sol_par = UNIT_OUT[key].squeeze()

            if len(UNIT.par_type_volfrac) == npartype:
                vol_par = self.get_vol_array(unit, 'particle') * UNIT.par_type_volfrac[ind]
            elif len(UNIT.par_type_volfrac) == nrad * npartype:
                par_type_volfrac = np.reshape(UNIT.par_type_volfrac, (nrad,npartype))
                vol_par = self.get_vol_array(unit, 'particle') * par_type_volfrac[:,ind]
            else:
                raise NotImplementedError

            par_radius = UNIT.par_radius[ind]
            npar = UNIT.discretization.npar[ind]

            if UNIT.discretization.par_disc_type == b'EQUIDISTANT_PAR':
                assert isinstance(par_radius, np.floating) or isinstance(par_radius, float)
                nShells = npar + 1 #Including r = 0
                rShells = [ par_radius * (n/npar) for n in range(nShells) ]
            else:
                print(UNIT.discretization.par_disc_type)
                raise NotImplementedError

            ## Reversed because of how it is in CADET
            vol_par_shells = np.array([ (r_out**3 - r_in**3)/par_radius**3 for r_out,r_in in pairwise(reversed(rShells)) ])

            # Integrated results per ncol x nrad cells
            # NOTE: Assumes npar is the last axis of sol_*
            sol_par = np.tensordot(sol_par, vol_par_shells, 1 )

            mass_par = sol_par * vol_par[np.newaxis, :]
            result.append(mass_par)

        # np.sum(list(map(lambda key: UNIT_OUT[key], keys)), axis=0)
        self.root.output.post[f'unit_{unit:03d}'].post_mass_par_all_partypes = np.sum(result, axis=0)

    def get_vol_rad(self, unit:int): 
        """ Return an array of shape (nrad,) with volumes of each radial port/zone """
        UNIT = self.root.input.model[f'unit_{unit:03d}']

        nrad = UNIT.discretization.nrad or 1 
        ncol = UNIT.discretization.ncol

        col_radius = UNIT.col_radius or np.sqrt(UNIT.cross_section_area / (np.pi))

        nShells = nrad + 1 #Including r = 0
        rShells = []

        if UNIT.discretization.radial_disc_type == b'EQUIVOLUME':
            rShells = [ col_radius * np.sqrt(n/nrad) for n in range(nShells) ]
        elif UNIT.discretization.radial_disc_type == b'EQUIDISTANT':
            rShells = [ col_radius * (n/nrad) for n in range(nShells) ]
        else: 
            rShells = [0.0, col_radius]

        dx = UNIT.col_length / ncol

        vol_rad = []
        for r_in, r_out, in zip(rShells[:-1], rShells[1:]+rShells[:0]):
            vol_rad.append(np.pi * (r_out**2 - r_in**2) * dx)

        return np.array(vol_rad)

    def get_vol_array(self, unit:int, output_type:str): 
        """ 
        Return an array of shape (ncol, nrad) but squeezed, containing volumes of each discretization volume by type. 
        output_type can be 'bulk', 'particle', 'solid', or 'total'
        """
        UNIT = self.root.input.model[f'unit_{unit:03d}']

        ncol = UNIT.discretization.ncol

        vol_rad = self.get_vol_rad(unit)

        # WARNING: if else to make it work for DPFRs
        col_porosity = np.array(UNIT.col_porosity) if UNIT.col_porosity is not None else 1.0
        par_porosity = np.array(UNIT.par_porosity) if UNIT.par_porosity is not None else 1.0

        col_radius = UNIT.col_radius or np.sqrt(UNIT.cross_section_area / (np.pi))

        total_volume = np.pi * col_radius**2 * UNIT.col_length
        assert np.isclose(np.sum(vol_rad) * ncol, total_volume, rtol=1e-6, atol=0.0)

        # WARNING: Assumes simple par_porosity
        if output_type == 'bulk': 
            factor = col_porosity
        elif output_type == 'particle': 
            factor = (1.0 - col_porosity) * par_porosity
        elif output_type == 'solid': 
            factor = (1.0 - col_porosity) * (1.0 - par_porosity)
        elif output_type == 'total': 
            factor = 1.0
        else: 
            raise ValueError("Bad output_type!")

        vol_rad_factor = vol_rad * factor
        vol_arr = np.stack([vol_rad_factor] * ncol, axis=0)

        return vol_arr.squeeze()

def new_run_and_eval(x, sim, parameters, objectives, name:Optional[str]=None, tempdir:Path=Path('temp'), store:bool=False): 
    """
    Run simulation -> Postprocess -> Evaluate objective scores
    """
    simulation = CadetSimulation()
    simulation.root = sim.root
    simulation.run_with_parameters(x, parameters, name, tempdir, store)

    # NOTE: This is a custom way to store postproc data in the hierarchy 
    # Postprocessed data is stored as "output.post.unit_001.post_internal_mass" for ex.
    # The first 'post' key separates it from cadet solutions
    # The final 'post_*' key is the name of the CadetSimulation method to run on the specified unit
    # Here, we run the postproc function specified. It is the responsibility of
    # the function to store the data at the right path
    obj_paths = list(set(obj.path for obj in objectives))
    for obj_path in obj_paths:
        path_split = obj_path.split('.')
        if path_split[1] == 'post':
            CadetSimulation.__dict__[path_split[3]](simulation, int(path_split[2].replace('unit_', '')))

    results = np.hstack(list(map(lambda obj: obj.evaluate(simulation), objectives)))

    return results

def new_run_sim_iter(index_x, sim, parameters, name:Optional[str]=None, tempdir:Path=Path('temp'), store:bool=False): 
    simulation = CadetSimulation()
    simulation.root = sim.root
    index, x = index_x
    simulation.run_with_parameters(x, parameters, f"{name}_{index}", tempdir, store)
    return simulation

def run_iter(index_x, sim, parameters, objectives, name:Optional[str]=None, tempdir:Path=Path('temp'), store:bool=False):
    simulation = CadetSimulation()
    simulation.root = sim.root
    index, x = index_x
    simulation.run_with_parameters(x, parameters, f"{name}_{index}", tempdir, store)
    obj_paths = list(set(obj.path for obj in objectives))
    for obj_path in obj_paths:
        path_split = obj_path.split('.')
        if path_split[1] == 'post':
            CadetSimulation.__dict__[path_split[3]](simulation, int(path_split[2].replace('unit_', '')))
    return simulation
