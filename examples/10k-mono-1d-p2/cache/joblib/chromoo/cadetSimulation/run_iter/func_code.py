# first line: 573
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
