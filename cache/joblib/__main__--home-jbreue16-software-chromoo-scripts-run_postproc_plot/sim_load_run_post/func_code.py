# first line: 13
@memory.cache
def sim_load_run_post():
    sim = CadetSimulation()
    sim.load_file('./long.poly2d.yaml')
    sim.save()
    sim.run_simulation()
    sim.load()
    return sim
