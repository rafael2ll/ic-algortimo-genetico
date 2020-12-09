from typing import List

import numpy as np
from numpy.random import default_rng

from ag.population import Population
from domain.cities import EuclideanCity
from domain.path.euclidean_extractor import natural_select


class PathPopulation(Population):
    def __init__(self):
        self.cities = []
        self.new_population = np.array([])

    def init(self, cities: List[EuclideanCity], nro_chromosomes=100):
        self.cities = cities
        self.new_population = np.array([
            np.random.permutation(range(len(self.cities))) for _ in list(range(nro_chromosomes))])
        print(self.new_population[:, 0])

    def get(self, idx) -> np.array:
        return self.new_population[idx]

    def get_pop(self) -> np.array:
        return self.new_population

    def add_offspring(self, offspring: np.array):
        print("offs", offspring)
        self.new_population = np.append(self.new_population, offspring, axis=0)

    def natural_selection(self):
        lowest_adaptive: List = natural_select(self.cities, self.new_population)
        self.new_population = np.delete(self.new_population, lowest_adaptive, axis=0)


if __name__ == '__main__':
    cities = [EuclideanCity(i, x[0], x[1]) for i, x in enumerate([(k, j) for k, j in zip(range(0, 10), range(0, 10))])]
    p = PathPopulation()
    p.init(cities, nro_chromosomes=6)
    print(p.new_population)
