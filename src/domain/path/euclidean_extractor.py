import math
import random
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
        distances = sorted(
            enumerate([sum(problem.get_weight(a, b) for a, b in zip(chromosome[0:], chromosome[1:])) for chromosome in
                       ciclic_pop]), key=lambda a: a[1])
        weights = [1] * len(distances)
        weights[0] = 100
        weights[1] = 100
        logger.debug(f"Distances: {distances}")
        logger.debug(f"Weights: {weights}")
        parents = random.choices(distances, weights=weights, k=1)
        while True:
            parent2 = random.choices(distances, weights=weights, k=1)[0]
            if parents[0] != parent2:
                parents.append(parent2)
                break
        logger.debug(f'Parents: {parents}')
        return np.array([population[parents[0][0]], population[parents[1][0]]])

    # Retorna os indices com menos adaptados


def natural_select(problem: tsplib95, population: np.array, die=0) -> List[int]:
    dist = Euclidean()
    distances = [sum(problem.get_weight(a, b) for a, b in zip(chromosome[0:], chromosome[1:])) for chromosome in
                 population]
    logger.debug(distances)
    lowest_list = []
    for d in range(die):
        lowest = max(enumerate(distances), key=lambda d: d[1])[0]
        distances[lowest] = -math.inf
        lowest_list.append(lowest)
    return lowest_list


if __name__ == '__main__':
    cities = [EuclideanCity(i, x[0], x[1]) for i, x in
              enumerate([(k, j) for k, j in zip(range(0, 10), range(0, 10))])]
    # p = PathPopulation(cities)
    # p.init(nro_chromosomes=6)
    # print(p.new_population)
    # print(EuclideanPathParentExtractor().extract_parent(cities, p))
