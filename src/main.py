import logging

import tsplib95
from joblib import Parallel, delayed

from ag import GA
from domain.crossovers import MPXCrossOver, VRCrossOver, PMXCrossOver, APCrossOver, ERCrossOver, OX2CrossOver
from domain.mutations import DM, EM, ISM, SIM, SM, IVM
from domain.path.euclidean_extractor import EuclideanPathParentExtractor
from domain.path.path_population import PathPopulation


def run_al(cross_over, mutation, problem):
    logging.warning(f" Running: {cross_over.__class__.__name__} x {mutation.__class__.__name__}")
    ag = GA()
    ag.set_population_class(PathPopulation())
    ag.set_parent_extractor_class(EuclideanPathParentExtractor())
    ag.set_crossover_class(cross_over)
    ag.set_mutation_class(mutation)
    distance, path = ag.perform(problem, 200)
    print(f" {cross_over.__class__.__name__} x {mutation.__class__.__name__}: Best Path[{distance}]: {path}")


if __name__ == '__main__':
    cross_overs = [APCrossOver(), PMXCrossOver(), MPXCrossOver(), VRCrossOver()]
    mutations = [DM(), EM(), ISM(), IVM(), SIM(), SM()]
    combinations = [(co, mt) for mt in mutations for co in cross_overs]
    problem = tsplib95.load('../data/gr24.tsp')
    print(list(problem.get_edges()))
    best_solution = tsplib95.load("../data/gr24.opt.tour")
    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s', level=logging.DEBUG)
    run_al(OX2CrossOver(), ISM(), problem)
    Parallel(n_jobs=len(mutations))(delayed(run_al)(comb[0], comb[1], problem) for comb in combinations)
