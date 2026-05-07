import copy
import random
import string
import subprocess
from pathlib import Path

from chromoo.utils import keystring_todict, deep_get, readChromatogram, readArray
from chromoo.scores import scores_dict
import numpy as np
import os

from cadet import Cadet
from addict import Dict

from ruamel.yaml import YAML

def load_file(fname):
    yaml=YAML(typ='safe')
    yaml.default_flow_style=False
    ext = Path(fname).suffix

    if ext == ".h5":
        simulation = loadh5(fname)
    elif ext == ".yaml" or ext == ".yml" :
        simulation = Cadet()
        cadetpath = subprocess.run(['which', 'cadet-cli'], capture_output=True, text=True ).stdout.strip()
        Cadet.cadet_path = cadetpath
        simulation.filename = fname.replace(ext, '.h5')
        simulation.root = Dict(yaml.load(Path(fname)))
    else:
        raise(RuntimeError('Invalid simulation file!'))

    return simulation


def loadh5(filename):
    """
        Load a cadet simulation file
    """

    cadetpath = subprocess.run(['which', 'cadet-cli'], capture_output=True, text=True ).stdout.strip()
    Cadet.cadet_path = cadetpath

    sim = Cadet()
    sim.filename = filename
    sim.load()

    return sim

def run_and_eval(x, sim, parameters, objectives, name=None, tempdir=Path('temp'), store=False ) -> list :
    sim = run_sim(x, sim, parameters, name=name, tempdir=tempdir, store=store)
    scores = evaluate_sim(sim, objectives)
    return scores


def run_sim(x, sim, parameters, name=None, tempdir=Path('temp'), store=False):
    """
        - run one simulation
        - calculate and return scores
    """
    newsim = copy.deepcopy(sim)

    if name:
        newsim.filename = name
    else:
        newsim.filename = tempdir.joinpath('temp' + ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6)) + '.h5')

    update_sim_parameters(newsim, x, parameters)

    newsim.save()

    try:
        return_information = newsim.run_simulation()
        if return_information.return_code != 0:
            print(f"{newsim.filename} failed with return code {return_information.return_code}")

    except subprocess.CalledProcessError as error:
        print(f"{newsim.filename} failed: {error.stderr.decode('utf-8').strip()}")
        print(f"Parameters: {x}\n")
        raise(RuntimeError("Simulation Failure"))

    newsim.load()

    if not store:
        os.remove(newsim.filename)

    return newsim

def run_sim_iter(index_x, sim, parameters, name='sim', tempdir=Path('temp'), store=False):
    """
        - run one simulation
        - calculate and return scores
        - Made for pool.map to use index as part of the filename
        TODO: Refactor with other run_sim
    """
    newsim = copy.deepcopy(sim)

    index = index_x[0]
    x = index_x[1]

    newsim.filename = tempdir.joinpath(f"{name}_{index:03d}.h5")

    update_sim_parameters(newsim, x, parameters)

    newsim.save()

    try:
        return_information = newsim.run_simulation()
        if return_information.return_code != 0:
            print(f"{newsim.filename} failed with return code {return_information.return_code}")
    except subprocess.CalledProcessError as error:
        print(f"{newsim.filename} failed: {error.stderr.decode('utf-8').strip()}")
        print(f"Parameters: {x}\n")
        raise(RuntimeError("Simulation Failure"))

    newsim.load()

    if not store:
        os.remove(newsim.filename)

    return newsim

def evaluate_sim(newsim, objectives):
    scores = []

    for obj in objectives:
        y = deep_get(newsim.root, obj.path)
        y = np.array(y).flatten()
        if obj.times: 
            y0 = readArray(obj.filename)
        else:
            _, y0 = readChromatogram(obj.filename)

        scores.append(scores_dict[obj.score](y0, y))

    return scores

def update_sim_parameters(sim, x, parameters):
    # For every parameter, generate a dictionary based on the path, and
    # update the simulation in a nested way
    # TODO: Probably a neater way to do this

    prev_len = 0
    for p in parameters:
        cur_len = p.length

        if p.type == 'element' :
            arr = sim.root
            for key in p.path.split('.'):
                arr = arr[key]
            value = x[prev_len : prev_len + cur_len]
            for idx_len,idx_arr in zip(range(cur_len), p.index):
                arr[idx_arr] = value[idx_len]
            cur_dict = keystring_todict(p.path, arr)

            ## NOTE: Hack to copy flowrate values within the connections matrix
            if p.copy_to_path: 
                arr2 = sim.root
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

        sim.root.update(cur_dict)
        prev_len += p.length
