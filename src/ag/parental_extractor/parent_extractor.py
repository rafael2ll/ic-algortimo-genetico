from abc import ABC, abstractmethod
from typing import Any


class ParentExtractor(ABC):
    @abstractmethod
    def extract_parent(self, data: Any, population: Any) -> Any:
        pass
