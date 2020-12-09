import numpy as np


class EuclideanCity:
    def __init__(self, idx, x, y):
        self.idx = idx
        self.x = float(x)
        self.y = float(y)

    def get_index(self):
        return self.idx

    def __sub__(self, other) -> np.array:
        return np.array([self.x, self.y]) - np.array([other.x, other.y])

    def __str__(self):
        return f"({self.x}, {self.y})"
