from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ner_model import NERModel
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
    def __init__(self, num_words: int, num_classes: int):
        self._model = BiLSTM(num_words, num_classes)
        self._loss = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)

    def train(
        self,
        features: List[List[int]],
        targets: List[List[int]],
        epochs: int,
        batch_size: int,
    ) -> None:
        features_tensor = torch.tensor(features, dtype=torch.long)
        targets_tensor = torch.tensor(targets, dtype=torch.long)
        dataset = TensorDataset(features_tensor, targets_tensor)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self._model.train()

        for epoch in range(epochs):
            for batch_features, batch_targets in dataloader:
                predictions = self._model(batch_features)
                predictions = predictions.view(-1, predictions.size(2))
                batch_targets = batch_targets.view(-1)

                loss = self._loss(predictions, batch_targets)
                loss.backward()
                self._optimizer.step()

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
