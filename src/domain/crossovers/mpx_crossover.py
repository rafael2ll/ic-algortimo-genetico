from random import randrange
from typing import Any

import numpy as np

from ag.crossovers import CrossOver


class MPXCrossOver(CrossOver):

    def cross(self, parents: Any, offspring_count: int = 1) -> Any:
        p1 = parents[0]
        p2 = parents[1]
        length = len(p1)

        if length > 1:
            num_elements = randrange(1, length // 2 + 1)
            start = randrange(0, length - num_elements)
            end = start + num_elements
            sub_arr = p1[start:end].tolist()
            s = set(sub_arr)

            for value in p2:
                if value not in s:
                    sub_arr.append(value)
                    s.add(value)

            return np.array([sub_arr])
        else:
            return np.array([p1])
