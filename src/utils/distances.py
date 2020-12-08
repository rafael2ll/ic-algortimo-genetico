import abc
from math import pi

import numpy as np


class Distance(abc.ABC):
    @abc.abstractmethod
    def calc(self, city_a, city_b) -> float:
        pass


class Euclidean(Distance):
    def calc(self, city_a: np.array, city_b: np.array) -> float:
        return np.linalg.norm(city_a - city_b)


class GeoCoord:
    def __init__(self, degrees=None, minutes=None):
        if degrees < 0:
            (self.degrees, self.minutes) = (degrees, -1 * minutes)
        else:
            (self.degrees, self.minutes) = (degrees, minutes)

    def toRadians(self):
        return (self.degrees + self.minutes / 60) * pi / 180.0


class GeoCity:
    def __init__(self, lat=None, lon=None):
        self.lat = lat.toRadians()
        self.lon = lon.toRadians()

    def coord_tuple(self):
        return (self.lat, self.lon)
