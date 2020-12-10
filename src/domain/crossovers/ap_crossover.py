from typing import Any

import numpy as np

from ag.crossovers import CrossOver


class APCrossOver(CrossOver):

    def cross(self, parents: Any, offspring_count: int = 1, more=None) -> Any:
        offspring = []
        already_in = set()
        for values in zip(*parents):
            for value in values:
                if value not in already_in:
                    offspring.append(value)
                    already_in.add(value)

        return np.array([offspring])
