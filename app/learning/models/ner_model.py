from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class NERModel(ABC):
    @classmethod
    def from_torch(cls, file_path: str) -> NERModel:
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        features: List[List[int]],
        targets: List[List[int]],
        epochs: int,
        batch_size: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def train_async(
        self,
        features: List[List[int]],
        targets: List[List[int]],
        epochs: int,
        batch_size: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, unlabeled_sentence: List[int]) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, file_path: str) -> None:
        raise NotImplementedError
