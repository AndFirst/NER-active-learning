import threading
from queue import Queue

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ner_model import NERModel


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
        # @TODO dodać prawdziwy padding idx
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

    def __init__(self, num_words: int, num_classes: int, learning_rate: float) -> None:
        super().__init__()
        self._model = BiLSTM(num_words, num_classes)
        self._loss = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        self._new_model = None
        self._lock = threading.Lock()
        self._training_queue = Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
