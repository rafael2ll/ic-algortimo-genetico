# %%

import logging

import pandas as pd
import tqdm
import tsplib95
from joblib import Parallel, delayed

from ag import GA
from domain.crossovers import VRCrossOver, PMXCrossOver, APCrossOver, CXCrossOver, OX1CrossOver, OX2CrossOver, \
    POSCrossOver, ERCrossOver
from domain.mutations import DM, EM, ISM, SIM, SM, IVM
from domain.path.euclidean_extractor import EuclideanPathParentExtractor
from domain.path.path_population import PathPopulation


# %%

def run_al(cross_over, mutation, problem):
    try:
        logging.warning(f" Running: {cross_over.__class__.__name__} x {mutation.__class__.__name__}")
        ag = GA()
        ag.set_population_class(PathPopulation())
        ag.set_parent_extractor_class(EuclideanPathParentExtractor())
        ag.set_crossover_class(cross_over)
        ag.set_mutation_class(mutation)
        distance, path = ag.perform(problem, 200)
        print(f" {cross_over.__class__.__name__} x {mutation.__class__.__name__}: Best Path[{distance}]: {path}")
        return cross_over.__class__.__name__, mutation.__class__.__name__, distance, path
    except Exception as e:
        logging.warning(f" Couldn't do the comb: {cross_over} x {mutation} : {e}")
        return cross_over.__class__.__name__, mutation.__class__.__name__, None, None


# %%

cross_overs = [APCrossOver(), CXCrossOver(), ERCrossOver(), OX1CrossOver(), OX2CrossOver(), PMXCrossOver(),
               POSCrossOver(), VRCrossOver()]

# %%

mutations = [DM(), EM(), ISM(), IVM(), SIM(), SM()]

# %%

combinations = tqdm.tqdm([(co, mt) for mt in mutations for co in cross_overs])

# %%

problem = tsplib95.load('../data/gr24.tsp')
best_solution = tsplib95.load("../data/gr24.opt.tour")

# %%

results = Parallel(n_jobs=len(mutations))(delayed(run_al)(comb[0], comb[1], problem) for comb in combinations)

# %%

collected = []
for (co, mt, dist, path) in results:
    value = list(filter(lambda a: a["mutation"] == mt, collected))
    if len(value) == 0:
        value = {"mutation": mt}
        collected.append(value)
    else:
        value = value[0]
    value[co] = dist

# %%

df = pd.DataFrame(collected)
df.to_csv("problem24.csv")

# %%

problem = tsplib95.load('../data/gr48.tsp')
best_solution = tsplib95.load("../data/gr48.opt.tour")

# %%

results = Parallel(n_jobs=len(mutations))(delayed(run_al)(comb[0], comb[1], problem) for comb in combinations)

# %%

collected = []
for (co, mt, dist, path) in results:
    value = list(filter(lambda a: a["mutation"] == mt, collected))
    if len(value) == 0:
        value = {"mutation": mt}
        collected.append(value)
    else:
        value = value[0]
    value[co] = dist

# %%

df = pd.DataFrame(collected)
df.to_csv("problem48.csv")
