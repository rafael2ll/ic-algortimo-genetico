from typing import List

import numpy as np

from ag.crossovers.crossover import CrossOver
from domain.cities import EuclideanCity
from domain.path.euclidean_extractor import EuclideanPathParentExtractor
from domain.path.path_population import PathPopulation
from domain.path.path_representation import PathRepresentation


# [4,5,6] -> [1,6,8]
# [1,6,8] -> [4,5,6]
#
# [8] -> [6] -> [6] ->[4]
def reallocate(parent, original, replacer, cut0, cut1):
    offspring = np.zeros(shape=parent.shape, dtype=int)
    print(f"initial offspring: {offspring}")
    offspring[cut0:cut1] = replacer
    print(f"replace offspring: {offspring}")
    offspring[0:cut0] = [parent[i] if parent[i] not in replacer else int(resolve(parent[i], original, replacer))
                         for i in
                         range(0, cut0)]
    print(f"first side offspring: {offspring}")
    for i in range(cut1, len(parent)):
        if parent[i] in replacer:
            print(f"[{i}]Resolving conflict: ")
            v = resolve(parent[i], original, replacer)
            print(f"Conflict resolved: {v}")
        else:
            print(f"[{i}]No conflict found")
            v = parent[i]
        offspring[i] = v

    print(f"second side offspring: {offspring}")

    return offspring


def resolve(v, original: np.array, replacer: np.array):
    print(f"v:{v}\to:{original}, r:{replacer}")
    idx = np.where(replacer == v)[0][0]
    print(f"Position  of {v}: {idx}")
    new_v = original[idx]
    if new_v not in replacer:
        print(f"Assuming {v} as {new_v}")
        return new_v
    else:
        return resolve(original[idx], original, replacer)


class PMXCrossOver(CrossOver):
    def cross(self, parents: List[PathRepresentation], offspring_count: int = 1) -> List[PathRepresentation]:
        p1, p2 = parents[0], parents[1]
        offspring = []
        gene_count = len(p1)
        cuts = np.random.randint(0, gene_count, size=2)
        cuts = np.sort(cuts)
        print(cuts)
        parent1_cut, parent2_cut = p1[cuts[0]:cuts[1]], p2[cuts[0]:cuts[1]]
        offspring.append(reallocate(p1, parent1_cut, parent2_cut, cuts[0], cuts[1]))
        offspring.append(reallocate(p2, parent2_cut, parent1_cut, cuts[0], cuts[1]))
        return np.array(offspring)


if __name__ == '__main__':
    cities = [EuclideanCity(i, x[0], x[1]) for i, x in
              enumerate([(k, j) for k, j in zip(range(0, 10), range(0, 10))])]
    p = PathPopulation(cities)
    p.init(nro_chromosomes=6)
    print(p.new_population)
    parents = EuclideanPathParentExtractor().extract_parent(cities, p)
    PMXCrossOver().cross(parents, offspring_count=2)
