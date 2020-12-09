import math
from typing import List

import numpy as np
import tsplib95

from ag.parental_extractor import ParentExtractor
from domain.cities import EuclideanCity
from domain.path.path_representation import PathRepresentation
from utils.distances import Euclidean
from utils.logger import get_logger

logger = get_logger(__name__)


class EuclideanPathParentExtractor(ParentExtractor):
    def extract_parent(self, problem: tsplib95, population: np.array) -> List[PathRepresentation]:
        dist = Euclidean()
        ciclic_pop = np.hstack((population, np.array([population[:, 0]]).T))
        distances = [sum(problem.get_weight(a, b) for a, b in zip(chromosome[0:], chromosome[1:])) for chromosome in
                     ciclic_pop]

        logger.debug(f'Distances: {distances}')
        parent1 = min(enumerate(distances), key=lambda d: d[1])[0]
        distances[parent1] = math.inf
        logger.debug(f'Distances: {distances}')
        parent2 = min(enumerate(distances), key=lambda d: d[1])[0]
        return [population[parent1], population[parent2]]


# Retorna os indices com menos adaptados
def natural_select(problem: tsplib95, population: np.array) -> List[int]:
    dist = Euclidean()
    distances = [sum(problem.get_weight(a, b) for a, b in zip(chromosome[0:], chromosome[1:])) for chromosome in
                 population]
    logger.debug(distances)
    lowest1 = max(enumerate(distances), key=lambda d: d[1])[0]
    distances[lowest1] = -math.inf
    lowest2 = max(enumerate(distances), key=lambda d: d[1])[0]
    return [lowest1, lowest2]


if __name__ == '__main__':
    cities = [EuclideanCity(i, x[0], x[1]) for i, x in
              enumerate([(k, j) for k, j in zip(range(0, 10), range(0, 10))])]
    # p = PathPopulation(cities)
    # p.init(nro_chromosomes=6)
    # print(p.new_population)
    # print(EuclideanPathParentExtractor().extract_parent(cities, p))
