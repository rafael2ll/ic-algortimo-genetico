import abc


class City(abc.ABC):
    @abc.abstractmethod
    def get_index(self):
        pass
