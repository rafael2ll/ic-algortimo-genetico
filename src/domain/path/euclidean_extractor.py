import math
from typing import List

import numpy as np

from ag.parental_extractor import ParentExtractor
from domain.cities import EuclideanCity
from domain.path.path_representation import PathRepresentation
from utils.distances import Euclidean


class EuclideanPathParentExtractor(ParentExtractor):
    def extract_parent(self, data: List[EuclideanCity], population: np.array) -> List[PathRepresentation]:
        dist = Euclidean()
        ciclic_pop = np.hstack((population, np.array([population[:, 0]]).T))
        distances = [sum(dist.calc(data[a], data[b]) for a, b in zip(chromosome[0:], chromosome[1:])) for chromosome in
                     ciclic_pop]

        print(f'Distances: {distances}')
        parent1 = min(enumerate(distances), key=lambda d: d[1])[0]
        distances[parent1] = math.inf
        print(f'Distances: {distances}')
        parent2 = min(enumerate(distances), key=lambda d: d[1])[0]
        return [population[parent1], population[parent2]]


# Retorna os indices com menos adaptados
def natural_select(data: List[EuclideanCity], population: np.array) -> List[int]:
    dist = Euclidean()
    distances = [sum(dist.calc(data[a], data[b]) for a, b in zip(chromosome[0:], chromosome[1:])) for chromosome in
                 population]
    print(distances)
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
