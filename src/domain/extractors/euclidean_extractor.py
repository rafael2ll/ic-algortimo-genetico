from typing import Any, List

from ag.parental_extractor import ParentExtractor
from domain.cities import EuclideanCity
from domain.path.path_population import PathPopulation
from utils.distances import Euclidean


class EuclideanPathParentExtractor(ParentExtractor):
    def extract_parent(self, data: List[EuclideanCity], population: PathPopulation, fitness):
        dist = Euclidean()
        distances = [sum(data[i] for i in chromosome) for chromosome in population.new_population]