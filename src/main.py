import random

from ag import GA
from domain.cities import EuclideanCity
from domain.path.dm_mutation import DMutation
from domain.path.euclidean_extractor import EuclideanPathParentExtractor
from domain.path.path_population import PathPopulation
from domain.path.pmx_crossover import PMXCrossOver
from utils.distances import Euclidean
import numpy as np

if __name__ == '__main__':
    cities = [EuclideanCity(i, random.randint(0, 1000), random.randint(0, 1000)) for i, x in
              enumerate([(k, j) for k, j in zip(range(0, 10), range(0, 10))])]
    ag = GA()
    ag.set_population_class(PathPopulation()) \
        .set_parent_extractor_class(EuclideanPathParentExtractor()) \
        .set_crossover_class(PMXCrossOver()).set_mutation_class(DMutation())
    ag.perform(cities, 8)

