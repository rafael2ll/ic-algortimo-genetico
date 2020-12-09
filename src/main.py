import random

from ag import GA
from domain.cities import EuclideanCity
from domain.crossovers.pmx_crossover import PMXCrossOver
from domain.mutations.dm_mutation import DMutation
from domain.path.euclidean_extractor import EuclideanPathParentExtractor
from domain.path.path_population import PathPopulation
import logging

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s', level=logging.DEBUG)
    cities = [EuclideanCity(i, random.randint(0, 1000), random.randint(0, 1000)) for i, x in
              enumerate([(k, j) for k, j in zip(range(0, 10), range(0, 10))])]
    ag = GA()
    ag.set_population_class(PathPopulation())
    ag.set_parent_extractor_class(EuclideanPathParentExtractor())
    ag.set_crossover_class(PMXCrossOver())
    ag.set_mutation_class(DMutation())
    ag.perform(cities, 200)
