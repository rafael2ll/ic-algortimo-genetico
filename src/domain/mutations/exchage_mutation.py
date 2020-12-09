from random import randrange, random
from typing import Any

import numpy as np

from ag.mutations import Mutation
from utils.logger import get_logger

logger = get_logger(__name__)


class EM(Mutation):
    def apply(self, population: Any, rate=0.01) -> Any:
        return np.array([self.mutate(chromo) if random() <= rate else chromo for chromo in population])

    def mutate(self, offspring):
        length = len(offspring)

        if length > 1:
            first = randrange(0, length)
            second = randrange(0, length)
            while second == first:
                second = randrange(0, length)

            aux = offspring[first]
            offspring[first] = offspring[second]
            offspring[second] = aux

        return offspring
