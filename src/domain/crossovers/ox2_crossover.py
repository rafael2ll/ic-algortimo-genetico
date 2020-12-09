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


def ox2(first: np.array, second: np.array, maxPos: int):
    size = len(first)
    off1 = np.full(size, -1)
    off2 = np.full(size, -1)
    fillOff1 = {}
    fillOff2 = {}
    posA = get_selected_positions(size, maxPos)
    posB = posA.copy()

    for pos in posA:
        a = first[pos]
        b = second[pos]
        fillOff1[b] = pos
        fillOff2[a] = pos
        logger.debug(f"Marked {b} as already fill for offspring 1")
        logger.debug(f"Marked {a} as already fill for offspring 2")

    i = 0
    while i < size:
        a = first[i]
        b = second[i]

        if a not in fillOff1:
            off1[i] = a
            logger.debug(f"off1[{i}] = {a} as {a} was not in offspring 1 yet and came from parent one")

        else:
            pos = posA.pop(0)
            value = -2
            for key in fillOff1:
                if fillOff1[key] == pos:
                    value = second[pos]
            off1[i] = value
            logger.debug(f"off1[{i}] = {value} as {value} came from parent two")

        if b not in fillOff2:
            off2[i] = b
            logger.debug(f"off2[{i}] = {b} as {b} was not in offspring 2 yet and came from parent two")

        else:
            pos = posB.pop(0)
            value = -2
            for key in fillOff2:
                if fillOff2[key] == pos:
                    value = first[pos]
            off2[i] = value
            logger.debug(f"off2[{i}] = {value} as {value} came from parent one")

        i = i + 1

    return off1, off2


class OX2CrossOver(CrossOver):
    def cross(self, parents: List[PathRepresentation], offspring_count: int = 1) -> List[PathRepresentation]:
        first_parent, second_parent = parents[0], parents[1]
        off1, off2 = ox2(first_parent, second_parent)
        return np.array([off1, off2])


if __name__ == '__main__':
    first = PathRepresentation(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    second = PathRepresentation(np.array([2, 4, 6, 8, 7, 5, 3, 1]))
    OX2CrossOver().cross([first, second], offspring_count=2)
