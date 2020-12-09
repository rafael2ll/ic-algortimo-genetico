import logging

import tsplib95

from ag import GA
from domain.crossovers.pmx_crossover import PMXCrossOver
from domain.mutations.dm_mutation import DMutation
from domain.path.euclidean_extractor import EuclideanPathParentExtractor
from domain.path.path_population import PathPopulation

if __name__ == '__main__':
    problem = tsplib95.load('../data/gr24.tsp')
    print(problem.get_weight(1, 3))
    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
    cities = problem
    ag = GA()
    ag.set_population_class(PathPopulation())
    ag.set_parent_extractor_class(EuclideanPathParentExtractor())
    ag.set_crossover_class(PMXCrossOver())
    ag.set_mutation_class(DMutation())
    ag.perform(cities, 200)
