from collections import Counter
from random import randrange, shuffle
from typing import Any

import numpy as np

from ag.crossovers import CrossOver


class VRCrossOver(CrossOver):
    def cross(self, parents: Any, offspring_count: int = 1) -> Any:
        length = len(parents)
        threshold = randrange(1, length + 1)
        already_in = set()
        arr = []
        has_none = []

        for i in range(parents[0].size):
            counter = Counter()
            for array in parents:
                counter.update([array[i]])
            most_commons = counter.most_common()
            most_common = most_commons[0]

            if most_common[1] >= threshold and most_common[0] not in already_in:
                arr.append(most_common[0])
                already_in.add(most_common[0])
            else:
                has_none.append([i, most_commons])
                arr.append(None)

        if has_none:
            for i, most_commons in has_none:
                value = None
                shuffle(most_commons)

                for v, _ in most_commons:
                    if v not in already_in:
                        value = v
                        already_in.add(v)
                        break

                if value is not None:
                    arr[i] = value
                else:
                    for array in parents:
                        for value in array:
                            if value not in already_in:
                                arr[i] = value
                                already_in.add(value)
                                break
        # mutation to avoid Nones
        for i in range(len(arr)):
            if arr[i] is None:
                arr[i] = [a for a in parents[0] if a not in arr][0]
        return np.array([arr])

#
# if __name__ == '__main__':
#     problem = tsplib95.load('../../../data/gr24.tsp')
#     p = PathPopulation()
#     p.init(nro_chromosomes=6, problem=problem)
#     print(p.new_population)
#     parents = EuclideanPathParentExtractor().extract_parent(problem, p.new_population)
#     print(VRCrossOver().cross(parents, offspring_count=2))
