from typing import List

import numpy as np

from ag.crossovers.crossover import CrossOver
from domain.cities import EuclideanCity
from domain.extractors.euclidean_extractor import EuclideanPathParentExtractor
from domain.path.path_population import PathPopulation
from domain.path.path_representation import PathRepresentation


def reallocate(parent, original, replacer, cut0, cut1):
    offspring = [0] * (len(parent) - 1)
    offspring[cut0:cut1] = replacer
    offspring[0:cut0] = [parent[i] if parent[i] not in replacer else resolve(parent[i], original, replacer) for i in
                         range(0, cut0)]
    offspring[cut1:-1] = [parent[i] if parent[i] not in replacer else resolve(parent[i], original, replacer) for i in
                          range(cut1, len(parent))]

    return offspring


def resolve(v, original: np.array, replacer: np.array):
    print(f"v:{v}\to:{original}, r:{replacer}")
    idx = np.where(replacer == v)[0][0]
    print(f"Position  of {v}: {idx}")
    if original[idx] not in replacer:
        return original[idx]
    else:
        resolve(original[idx], replacer, original)


class PMXCrossOver(CrossOver):
    def cross(self, parents: List[PathRepresentation], offspring_count: int = 1) -> List[PathRepresentation]:
        p1, p2 = parents[0], parents[1]
        gene_count = len(p1)
        cuts = np.random.randint(0, gene_count, size=2)
        cuts = np.sort(cuts)
        print(cuts)
        parent1_cut, parent2_cut = p1[cuts[0]:cuts[1]], p2[cuts[0]:cuts[1]]
        print(reallocate(p1, parent1_cut, parent2_cut, cuts[0], cuts[1]))


if __name__ == '__main__':
    cities = [EuclideanCity(i, x[0], x[1]) for i, x in
              enumerate([(k, j) for k, j in zip(range(0, 10), range(0, 10))])]
    p = PathPopulation(cities)
    p.init(nro_chromosomes=6)
    print(p.new_population)
    parents = EuclideanPathParentExtractor().extract_parent(cities, p)
    PMXCrossOver().cross(parents, offspring_count=2)
