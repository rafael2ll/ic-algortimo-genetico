from random import randrange, random
from typing import Any

import numpy as np

from ag.mutations import Mutation
from utils.logger import get_logger

logger = get_logger(__name__)


class SIM(Mutation):
    def apply(self, population: Any, rate=0.01) -> Any:
        return np.array([self.mutate(chromo) if random() <= rate else chromo for chromo in population])

    def mutate(self, offspring):
        length = len(offspring)

        if length > 2:
            num_elements = randrange(2, length)
            start = randrange(0, length - num_elements)
            end = start + num_elements
            return np.concatenate((offspring[0:start], np.flip(offspring[start:end]), offspring[end:]))
        else:
            return np.flip(offspring)
