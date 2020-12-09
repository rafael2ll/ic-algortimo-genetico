from random import randrange, random
from typing import Any

import numpy as np

from ag.mutations import Mutation
from utils.logger import get_logger

logger = get_logger(__name__)


class IVM(Mutation):
    def apply(self, population: Any, rate=0.01) -> Any:
        return np.array([self.mutate(chromo) if random() <= rate else chromo for chromo in population])

    def mutate(self, offspring):
        length = offspring.size
        if length > 1:
            num_elements = randrange(1, length)
            start = randrange(0, length - num_elements)
            end = start + num_elements

            return np.insert(np.concatenate((offspring[0:start], offspring[end:])), start,
                             self._flip(offspring[start:end]))
        else:
            return offspring

    def _flip(self, arr: np.array):
        np.flip(arr)
        return arr
