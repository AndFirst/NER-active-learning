import asyncio
import copy
import threading
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from .ner_model import NERModel
from torch.utils.data import TensorDataset, DataLoader


class BiLSTM(nn.Module):
    def __init__(
        self,
        num_words,
        num_classes,
        embedding_dim=100,
        lstm_units=100,
        dropout=0.1,
    ):
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding(num_words, embedding_dim, padding_idx=0)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim,
            lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(lstm_units * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        embedded = self.bn(embedded.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_output, _ = self.lstm(embedded)
        logits = self.linear(lstm_output)
        probabilities = F.softmax(logits, dim=2)
        return probabilities


class BiLSTMClassifier(NERModel):

    def __init__(
        self, num_words: int, num_classes: int, learning_rate: float
    ) -> None:
        self._model = BiLSTM(num_words, num_classes)
        self._loss = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=learning_rate
        )
        self._new_model = None
        self._lock = threading.Lock()

    async def train(
        self,
        features: List[List[int]],
        targets: List[List[int]],
        epochs: int,
        batch_size: int,
    ) -> None:
        # Copy the model and initialize the new model for training
        with self._lock:
            self._new_model = copy.deepcopy(self._model)
        optimizer = optim.Adam(
            self._new_model.parameters(),
            lr=self._optimizer.param_groups[0]["lr"],
        )

        features_tensor = torch.tensor(features, dtype=torch.long)
        targets_tensor = torch.tensor(targets, dtype=torch.long)
        dataset = TensorDataset(features_tensor, targets_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self._new_model.train()

        for epoch in range(epochs):
            for batch_features, batch_targets in dataloader:
                optimizer.zero_grad()
                predictions = self._new_model(batch_features)
                predictions = predictions.view(-1, predictions.size(2))
                batch_targets = batch_targets.view(-1)

                loss = self._loss(predictions, batch_targets)
                loss.backward()
                optimizer.step()

        # Swap the new model with the old one
        with self._lock:
            self._model = self._new_model
            self._new_model = None
            print("swapped model")

    def train_async(
        self,
        features: List[List[int]],
        targets: List[List[int]],
        epochs: int,
        batch_size: int,
    ) -> None:
        asyncio.run(
            self._train_async_helper(features, targets, epochs, batch_size)
        )

    async def _train_async_helper(
        self,
        features: List[List[int]],
        targets: List[List[int]],
        epochs: int,
        batch_size: int,
    ) -> None:
        await self.train(features, targets, epochs, batch_size)

    def predict(self, unlabeled_sentence: List[int]) -> List[int]:
        features = torch.tensor(
            [
                unlabeled_sentence,
            ],
            dtype=torch.long,
        )

        self._model.eval()

        with torch.no_grad():
            predictions = np.argmax(self._model(features).cpu(), axis=-1)

        return predictions[0].tolist()

    def reset(self) -> None:
        raise NotImplementedError

    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)

    def load_weights(self, file_path: str):
        state_dict = torch.load(file_path)
        self._model.load_state_dict(state_dict)
