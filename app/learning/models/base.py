from __future__ import annotations

import copy
import logging
import threading
from abc import ABC
from queue import Queue
from typing import List, Tuple, Dict

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader


class NERModel(ABC):
    def __init__(self):
        self._model = None
        self._loss = None
        self._optimizer = None
        self._new_model = None
        self._lock = None
        self._training_queue = Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _train_model(
        self,
        features: List[List[int]],
        targets: List[List[int]],
        epochs: int,
        batch_size: int,
        class_weights: List[float] = None,
    ) -> None:
        # Copy the model and initialize the new model for training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with self._lock:
            self._new_model = copy.deepcopy(self._model).to(device)
        optimizer = optim.Adam(
            self._new_model.parameters(),
            lr=self._optimizer.param_groups[0]["lr"],
        )

        features_tensor = torch.tensor(features, dtype=torch.long).to(device)
        targets_tensor = torch.tensor(targets, dtype=torch.long).to(device)
        dataset = TensorDataset(features_tensor, targets_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self._new_model.train()

        if class_weights is None:
            self._loss = CrossEntropyLoss()
        else:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
            self._loss = CrossEntropyLoss(weight=class_weights_tensor)

        for epoch in range(epochs):
            for batch_features, batch_targets in dataloader:
                optimizer.zero_grad()
                predictions = self._new_model(batch_features)
                predictions = predictions.view(-1, predictions.size(2))
                batch_targets = batch_targets.view(-1)

                loss = self._loss(predictions, batch_targets)
                loss.backward()
                optimizer.step()

        with self._lock:
            self._model = self._new_model.to("cpu")
            self._new_model = None
            logging.info("Swapped model.")

    def train_async(
        self,
        features: List[List[int]],
        targets: List[List[int]],
        epochs: int,
        batch_size: int,
        class_weights: List[float] = None,
    ) -> None:
        self._training_queue.put((features, targets, epochs, batch_size, class_weights))

    def _worker(self) -> None:
        while True:
            features, targets, epochs, batch_size, class_weights = self._training_queue.get()
            self._train_model(features, targets, epochs, batch_size, class_weights)
            self._training_queue.task_done()

    def predict_with_confidence(self, unlabeled_sentence: List[int]) -> Tuple[List[int], List[float]]:
        features = torch.tensor(
            [
                unlabeled_sentence,
            ],
            dtype=torch.long,
        )

        self._model.eval()

        with torch.no_grad():
            probabilities = self._model(features).cpu()
            max_values, predictions = torch.max(probabilities, dim=-1)
            confidence_scores = max_values

        return predictions[0].tolist(), confidence_scores[0].tolist()

    def save(self, path: str) -> None:
        if self._model is None:
            raise ValueError("Model must be initialized before saving.")
        if self._optimizer is None:
            raise ValueError("Optimizer must be initialized before saving.")
        if self._loss is None:
            raise ValueError("Loss function must be initialized before saving.")

        model_state = {
            "model_state_dict": self._model.state_dict(),
            "optimizer_name": type(self._optimizer).__name__,
            "optimizer_state_dict": self._optimizer.state_dict(),
            "loss_name": type(self._loss).__name__,
            "loss_state_dict": (self._loss.state_dict() if hasattr(self._loss, "state_dict") else None),
        }
        torch.save(model_state, path)

    def load_weights(self, file_path: str) -> None:
        if self._model is None:
            raise ValueError("Model must be initialized before loading weights.")

        state_dict = torch.load(file_path)

        self._model.load_state_dict(state_dict["model_state_dict"])

        if "optimizer_state_dict" in state_dict and "optimizer_name" in state_dict:
            optimizer_class = getattr(optim, state_dict["optimizer_name"])
            self._optimizer = optimizer_class(self._model.parameters())
            self._optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        else:
            self._optimizer = optim.Adam(self._model.parameters())

        if "loss_state_dict" in state_dict and "loss_name" in state_dict:
            loss_class = getattr(torch.nn, state_dict["loss_name"])
            self._loss = loss_class()
            if state_dict["loss_state_dict"] is not None:
                state_dict["loss_state_dict"].pop("weight", None)
                self._loss.load_state_dict(state_dict["loss_state_dict"])
        else:
            self._loss = CrossEntropyLoss()

    def validate_torch_model(self, num_words: int, num_classes: int) -> None:
        layers = list(self._model.children())
        embedding_size = layers[0].num_embeddings
        if embedding_size != num_words:
            raise ValueError(f"Expected embedding size {num_words}, but got {embedding_size}.")

        out_features = layers[-1].out_features
        if out_features != num_classes:
            raise ValueError(f"Expected output size {num_classes}, but got {out_features}.")

    def evaluate_metrics(self, features: List[List[int]], targets: List[List[int]]) -> Dict[str, float | int]:
        """
        Evaluates the model's performance metrics on a given dataset.

        :param features: The features of the dataset.
        :type features: List[List[int]]
        :param targets: The true labels of the dataset.
        :type targets: List[List[int]]
        :return: A dictionary containing the accuracy, precision, recall, and F1 score of the model.
        :rtype: Dict[str, float]
        """
        # Convert the features and targets to tensors and pad sequences
        features_tensor = pad_sequence(
            [torch.tensor(f, dtype=torch.long) for f in features],
            batch_first=True,
        )
        targets_tensor = pad_sequence(
            [torch.tensor(t, dtype=torch.long) for t in targets],
            batch_first=True,
        )

        # Get the model's predictions
        self._model.eval()
        with torch.no_grad():
            probabilities = self._model(features_tensor).cpu()
            _, predictions = torch.max(probabilities, dim=-1)

        # Flatten the predictions and targets
        predictions_flat = predictions.view(-1).tolist()
        targets_flat = targets_tensor.view(-1).tolist()

        # Calculate the metrics
        accuracy = accuracy_score(targets_flat, predictions_flat)
        precision = precision_score(targets_flat, predictions_flat, average="weighted", zero_division=0)
        recall = recall_score(targets_flat, predictions_flat, average="weighted", zero_division=0)
        f1 = f1_score(targets_flat, predictions_flat, average="weighted", zero_division=0)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
