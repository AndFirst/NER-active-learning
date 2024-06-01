import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(
            self,
            num_words=100_000,
            num_classes=5,
            embedding_dim=100,
            lstm_units=100,
            dropout=0.1,
    ):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(num_words, embedding_dim)
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
