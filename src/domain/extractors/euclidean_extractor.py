from typing import List

from ag.parental_extractor import ParentExtractor
from domain.cities import EuclideanCity
from domain.path.path_population import PathPopulation
from domain.path.path_representation import PathRepresentation
from utils.distances import Euclidean


class EuclideanPathParentExtractor(ParentExtractor):
    def extract_parent(self, data: List[EuclideanCity], population: PathPopulation) -> List[PathRepresentation]:
        dist = Euclidean()
        distances = [sum(dist.calc(data[a], data[b]) for a, b in zip(chromosome[0:], chromosome[1:])) for chromosome in
                     population.new_population]
        print(distances)
        parent1 = min(enumerate(distances), key=lambda d: d[1])[0]
        distances.pop(parent1)
        parent2 = min(enumerate(distances), key=lambda d: d[1])[0]
        return [population.get(parent1), population.get(parent2)]


if __name__ == '__main__':
    cities = [EuclideanCity(i, x[0], x[1]) for i, x in
              enumerate([(k, j) for k, j in zip(range(0, 10), range(0, 10))])]
    p = PathPopulation(cities)
    p.init(nro_chromosomes=6)
    print(p.new_population)
    print(EuclideanPathParentExtractor().extract_parent(cities, p))
