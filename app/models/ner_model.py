from __future__ import annotations

from abc import ABC
from typing import List


class NERModel(ABC):
    @classmethod
    def from_torch(cls, file_path: str) -> NERModel:
        raise NotImplementedError

    def train(
        self,
        features: List[List[int]],
        targets: List[List[int]],
        epochs: int,
        batch_size: int,
    ) -> None:
        raise NotImplementedError

    def predict(self, unlabeled_sentence: List[int]) -> List[int]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
