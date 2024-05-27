from abc import ABC
from typing import List, TypeAlias

Sentences: TypeAlias = List[List[str]]


class DataPersistenceStrategy(ABC):
    def load(self) -> Sentences:
        raise NotImplementedError

    def save(self, sentences: Sentences) -> None:
        raise NotImplementedError
