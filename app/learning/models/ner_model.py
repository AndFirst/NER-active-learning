from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch


class NERModel(ABC):
    def __init__(self):
        self._model = None
        self._loss = None
        self._optimizer = None
        self._new_model = None
        self._lock = None

    @classmethod
    def from_torch(cls, file_path: str) -> NERModel:
        raise NotImplementedError

    #
    # @abstractmethod
    # def train(
    #     self,
    #     features: List[List[int]],
    #     targets: List[List[int]],
    #     epochs: int,
    #     batch_size: int,
    # ) -> None:
    #     raise NotImplementedError

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

    def validate_torch_model(self, num_words: int, num_classes: int) -> None:
        layers = list(self._model.children())
        if not isinstance(layers[0], torch.nn.Embedding):
            raise ValueError("The first layer is not an Embedding layer.")
        if layers[0].num_embeddings != num_words:
            raise ValueError(
                f"Expected embedding size {num_words}, but got {layers[0].num_embeddings}."
            )

        if not isinstance(layers[-1], torch.nn.Linear):
            raise ValueError("The last layer is not a Linear layer.")
        if layers[-1].out_features != num_classes:
            raise ValueError(
                f"Expected output size {num_classes}, but got {layers[-1].out_features}."
            )
