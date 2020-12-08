from typing import List

import numpy as np
from numpy.random import default_rng

from ag.population import Population
from domain.cities import EuclideanCity


class PathPopulation(Population):
    def __init__(self, cities: List[EuclideanCity]):
        self.cities = cities
        self.new_population = np.array([])

    def init(self, nro_chromosomes=100):
        self.new_population = np.array([
            np.random.permutation(range(len(self.cities))) for _ in list(range(nro_chromosomes))])
        print(self.new_population[:, 0])
        self.new_population = np.hstack((self.new_population, np.array([self.new_population[:, 0]]).T))

    def get(self, idx) -> np.array:
        return self.new_population[idx]


if __name__ == '__main__':
    p = PathPopulation([EuclideanCity(i, x[0], x[1]) for i, x in
                        enumerate([(k, j) for k, j in zip(range(0, 10), range(0, 10))])])
    p.init(nro_chromosomes=6)
    print(p.new_population)
