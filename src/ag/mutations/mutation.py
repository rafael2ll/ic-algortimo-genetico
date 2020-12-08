from abc import ABC, abstractmethod
from typing import Any


class Mutation(ABC):
    @abstractmethod
    def apply(self, offspring: Any) -> Any:
        pass
