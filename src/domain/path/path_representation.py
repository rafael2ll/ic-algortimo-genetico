from typing import List

import numpy as np

from ag.representation import Representation


class PathRepresentation(Representation):
    def __init__(self, chromosome: List, population: List):
        self.path_idx = np.array(list(map(lambda a: population.index(a), chromosome)))

    def get(self):
        return self.path_idx
