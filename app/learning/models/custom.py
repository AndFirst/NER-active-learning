import threading
from typing import List

import torch
import torch.optim as optim
import torch.nn as nn

from app.learning.models.ner_model import NERModel


class CustomModel(NERModel):
    def __init__(
        self, num_words: int, num_classes: int, learning_rate: float
    ) -> None:
        self._model = None
        self._new_model = None
        self._optimizer = None
        self._loss = None
        self._lock = threading.Lock()

    def train(
        self,
        features: List[List[int]],
        targets: List[List[int]],
        epochs: int,
        batch_size: int,
    ) -> None:
        pass

    async def train_async(
        self,
        features: List[List[int]],
        targets: List[List[int]],
        epochs: int,
        batch_size: int,
    ) -> None:
        pass

    def predict(self, unlabeled_sentence: List[int]) -> List[int]:
        pass

    def save(self, path: str) -> None:
        checkpoint = {
            "model": self._model,
            "optimizer_state_dict": self._optimizer.state_dict(),
            "optimizer_class": self._optimizer.__class__.__name__,
            "optimizer_params": self._optimizer.defaults,
            "loss_function": self._loss.__class__.__name__,
        }
        torch.save(checkpoint, path)

    def load_weights(self, path: str) -> None:
        checkpoint = torch.load(path)

        self._model = checkpoint["model"]
        try:
            optimizer_class = getattr(optim, checkpoint["optimizer_class"])
            self._optimizer = optimizer_class(
                self._model.parameters(),
                **checkpoint["optimizer_params"],
            )
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (AttributeError, KeyError):
            self._optimizer = optim.Adam(self._model.parameters())

        try:
            loss_function_class = getattr(nn, checkpoint["loss_function"])
            self._loss = loss_function_class()
        except (AttributeError, KeyError):
            self._loss = nn.CrossEntropyLoss()
