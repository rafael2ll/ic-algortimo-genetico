from typing import List

import numpy as np
from ag.crossovers.crossover import CrossOver
from domain.cities import EuclideanCity
from domain.path.euclidean_extractor import EuclideanPathParentExtractor
from domain.path.path_population import PathPopulation
from domain.path.path_representation import PathRepresentation
from utils.logger import get_logger

logger = get_logger(__name__)


def get_selected_positions(size: int, maxPos: int):
    pos = [i for i in range(size)]
    selectedPos = []
    i = 0

    while i < maxPos:
        size = len(pos)
        index = np.random.randint(low=0, high=size, dtype=int)
        selectedPos.append(pos[index])
        pos.pop(index)
        i = i + 1

    selectedPos.sort()
    logger.debug(f"Selected random positions are: {selectedPos}")
    return selectedPos


def populate_initial_values(parent: np.array, positions: list, off: np.array, fill_off: dict):
    while len(positions) > 0:
        actual_pos = positions.pop()
        elem = parent[actual_pos]
        off[actual_pos] = elem
        fill_off[elem] = True
        logger.debug(f"Initial value {elem} for offspring")


def copy_remaining_values(parent: np.array, off: np.array, fill_off: dict, size: int):
    i = 0
    actual_pos = 0
    for elem in off:
        if elem == -1:
            actual = parent[i]
            while True and i < size:
                actual = parent[i]
                if actual not in fill_off:
                    break
                else:
                    i = i + 1
            off[actual_pos] = actual
            fill_off[actual] = True
            logger.debug(f"Set value {actual} for offspring")

        actual_pos = actual_pos + 1


def pos(first: np.array, second: np.array, maxPos: int):
    size = len(first)
    off1 = np.full(size, -1)
    off2 = np.full(size, -1)
    pos_a = get_selected_positions(size, maxPos)
    pos_b = pos_a.copy()
    fill_off1 = {}
    fill_off2 = {}

    logger.debug("Populate initial values for offspring 1")
    populate_initial_values(second, pos_a, off1, fill_off1)
    logger.debug("Populate initial values for offspring 2")
    populate_initial_values(first, pos_b, off2, fill_off2)

    logger.debug("Copy remaining values for offspring 1")
    copy_remaining_values(first, off1, fill_off1, size)
    logger.debug("Copy remaining values for offspring 2")
    copy_remaining_values(second, off2, fill_off2, size)

    return off1, off2


class POSCrossOver(CrossOver):
    def cross(self, parents: List[PathRepresentation], offspring_count: int = 1, more=None) -> List[PathRepresentation]:
        first_parent, second_parent = parents[0], parents[1]
        max_count = np.random.randint(low=0, high=len(first_parent), dtype=int)
        off1, off2 = pos(first_parent, second_parent, max_count)
        return np.array([off1, off2])


if __name__ == '__main__':
    first = PathRepresentation(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    second = PathRepresentation(np.array([2, 4, 6, 8, 7, 5, 3, 1]))
    POSCrossOver().cross([first, second], offspring_count=2)
