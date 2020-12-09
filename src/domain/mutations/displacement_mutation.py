import random
from typing import Any

import numpy as np

from ag.mutations import Mutation
from utils.logger import get_logger

logger = get_logger(__name__)


class DM(Mutation):
    def apply(self, population: Any, rate=0.01) -> Any:
        return np.array([self.mutate(chromo) if random.random() <= rate else chromo for chromo in population])

    def mutate(self, offspring):
        logger.debug("Mutated")
        if len(offspring) > 1:
            cuts = np.random.randint(0, len(offspring), size=2)
            cuts = np.sort(cuts)
            cutted = offspring[cuts[0]:cuts[1]]
            one = np.concatenate((offspring[0:cuts[0]], offspring[cuts[1]:]), axis=None)
            insert_point = np.random.randint(0, len(one), size=1)[0]
            logger.debug(f"{cuts}\t {cutted}\t{one}\t{insert_point}")
            return np.concatenate((one[0:insert_point], cutted, one[insert_point:]), axis=None)
        else:
            return offspring
