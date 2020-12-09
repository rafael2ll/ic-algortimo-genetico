from typing import List

import numpy as np
from ag.crossovers.crossover import CrossOver
from domain.cities import EuclideanCity
from domain.path.euclidean_extractor import EuclideanPathParentExtractor
from domain.path.path_population import PathPopulation
from domain.path.path_representation import PathRepresentation
from utils.logger import get_logger

logger = get_logger(__name__)


def has_next_node(current_node: int, edge_map: dict, visited: dict):
    possibles = []
    for node in edge_map[current_node]:
        if node not in visited: possibles.append(node)
    if len(possibles) == 0:
        r = (False, None)
        logger.debug(f"Current node {current_node} doesn't have next node, result = {r} ")
        return r
    else:
        r = True, possibles
        logger.debug(f"Current node {current_node} has next node, result = {r}")
        return r


def choose_next_node(current_node: int, edge_map: dict, possibles: list, visited: dict):
    count_neighboors = []
    for node in possibles:
        count = len(edge_map[node]) - 1
        count_neighboors.append((node, count))

    min_neighboors = min(count_neighboors, key=lambda x: x[1])[1]
    min_choices = []
    logger.debug(
        f"Possibles neighboors {possibles} for current node {current_node} has the following count neighboors {count_neighboors}")
    logger.debug(f"The minimum count neighboors is {min_neighboors}")
    for pair in count_neighboors:
        if pair[1] == min_neighboors and pair[0] not in visited:
            min_choices.append(pair)

    if len(min_choices) == 0:
        logger.debug(f"Neighboors for current node {current_node} has min value equal to 0")
        return None
    else:
        index = np.random.randint(low=0, high=len(min_choices), dtype=int)
        logger.debug(f"The next node choice is node {min_choices[index][0]}")
        return min_choices[index][0]


def has_random_next_node(parent: np.array, visited: dict):
    # Get next random node if possible for ER
    possibles = []
    for node in parent:
        if node not in visited:
            possibles.append(node)

    if len(possibles) == 0:
        r = (False, None)
        logger.debug(f"All nodes were already visited, result = {r}")
        return r
    else:
        index = np.random.randint(low=0, high=len(possibles), dtype=int)
        r = (True, possibles[index])
        logger.debug(f"Node {possibles[index]} was the random choice as next node, result = {r}")
        return r


def random_choose_first_node(parent_one: np.array, parent_two: np.array):
    choice = np.random.randint(low=0, high=2, dtype=int)
    if choice == 0:
        logger.debug("The first node is from parent one, choice = {parent_one[0]}")
        return parent_one[0]
    else:
        logger.debug(f"The first node is from parent two, choice = {parent_two[0]}")
        return parent_two[0]


def er(first, second, edge_map):
    # Genetic Edge Recombination Crossover (ER)
    current_node = random_choose_first_node(first, second)
    off = []
    visited = {}

    while True:
        visited[current_node] = True
        off.append(current_node)
        has_next, possibles = has_next_node(current_node, edge_map, visited)

        if has_next:
            next_node = choose_next_node(current_node, edge_map, possibles, visited)
            logger.debug(f"Current node is {current_node}, next node will be {next_node}, visited = {visited}")
            current_node = next_node

        else:
            has_random_next, next_random = has_random_next_node(first, visited)
            if has_random_next:
                logger.debug("Random next node will be chosen")
                current_node = next_random
            else:
                logger.debug("Finish ER")
                break

    return off


class ERCrossOver(CrossOver):
    def cross(self, parents: List[PathRepresentation], offspring_count: int = 1) -> List[PathRepresentation]:
        # TODO: Get edge map somehow for the current parents
        edge_map = {1: [2, 6, 3, 5], 2: [1, 3, 4, 6], 3: [2, 4, 1], 4: [3, 5, 2], 5: [4, 6, 1], 6: [1, 5, 2]}
        first_parent = np.array([1, 2, 3, 4, 5, 6])
        second_parent = np.array([2, 4, 3, 1, 5, 6])
        # TODO: end
        off = er(first_parent, second_parent, edge_map)
        return np.array(off)


if __name__ == '__main__':
    first = PathRepresentation(np.array([1, 2, 3, 4, 5, 6]))
    second = PathRepresentation(np.array([2, 4, 3, 1, 5, 6]))
    ERCrossOver().cross([first, second], offspring_count=2)
