from typing import List

import numpy as np
import random as r
from ag.crossovers.crossover import CrossOver
from domain.cities import EuclideanCity
from domain.path.euclidean_extractor import EuclideanPathParentExtractor
from domain.path.path_population import PathPopulation
from domain.path.path_representation import PathRepresentation
from utils.logger import get_logger

logger = get_logger(__name__)


# CX
def get_random_positions(size: int):
    positions = [i for i in range(size)]
    r.shuffle(positions)
    logger.debug(f"Random positions are {positions}")
    return positions


def cx(first: np.array, second: np.array):
    choice = np.random.randint(low=0, high=2, dtype=int)
    size = len(first)
    positions = get_random_positions(size)
    fill_off = {}
    off = np.full(size, -1)

    while len(positions) > 0:
        pos = positions.pop()
        a = first[pos]
        b = second[pos]

        if a not in fill_off and b not in fill_off:
            if choice == 0:
                off[pos] = a
                fill_off[a] = True
                logger.debug(f"Deal with tie by choosing parent one and off[{pos}] = {a}")
            else:
                off[pos] = b
                fill_off[b] = True
                logger.debug(f"Deal with tie by choosing parent two and off[{pos}] = {b}")
        elif a not in fill_off:
            off[pos] = a
            fill_off[a] = True
            logger.debug(f"Set {a} for off[{pos}] = {a}")

        elif b not in fill_off:
            off[pos] = b
            fill_off[b] = True
            logger.debug(f"Set {b} for off[{pos}] = {b}")
        else:
            logger.debug(
                f"This case shouldn't happen but happened for pos = {pos}, a = {a}, b = {b} and fill_off = {fill_off}")
    return off


class CXCrossOver(CrossOver):
    def cross(self, parents: List[PathRepresentation], offspring_count: int = 1) -> List[PathRepresentation]:
        first_parent, second_parent = parents[0], parents[1]
        off = cx(first_parent, second_parent)
        return np.array([off])


if __name__ == '__main__':
    first = PathRepresentation(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    second = PathRepresentation(np.array([2, 4, 6, 8, 7, 5, 3, 1]))
    CXCrossOver().cross([first, second], offspring_count=2)
