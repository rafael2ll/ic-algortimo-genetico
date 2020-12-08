from abc import ABC, abstractmethod

from numpy import array


class Population(ABC):
    @abstractmethod
    def get(self, idx) -> array:
        pass

    @abstractmethod
    def init(self, nro_chromosomes=100):
        pass
