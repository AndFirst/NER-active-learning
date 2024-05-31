from __future__ import annotations

import copy
import logging
import threading
from queue import Queue
from typing import List, Tuple, Dict

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader


class NERModel:
    """
    A class used to represent a Named Entity Recognition (NER) Model.
    """

    def __init__(self):
        self._model = None
        self._loss = None
        self._optimizer = None
        self._new_model = None
        self._lock = threading.Lock()
        self._training_queue = Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _initialize_new_model(self) -> None:
        """
        Initializes a new model for training.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with self._lock:
            self._new_model = copy.deepcopy(self._model).to(device)
        self._optimizer = optim.Adam(
            self._new_model.parameters(),
            lr=self._optimizer.param_groups[0]["lr"],
        )

    def _prepare_data(self, features: List[List[int]], targets: List[List[int]], batch_size: int) -> DataLoader:
        """
        Prepares the data for training.

        :param features: The features to be used for training.
        :type features: List[List[int]]
        :param targets: The targets to be used for training.
        :type targets: List[List[int]]
        :param batch_size: The batch size to be used for training.
        :type batch_size: int
        :return: A DataLoader object.
        :rtype: DataLoader
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features_tensor = torch.tensor(features, dtype=torch.long).to(device)
        targets_tensor = torch.tensor(targets, dtype=torch.long).to(device)
        dataset = TensorDataset(features_tensor, targets_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def _set_loss_function(self, class_weights: List[float]) -> None:
        """
        Sets the loss function for training.

        :param class_weights: The class weights to be used for training.
        :type class_weights: List[float]
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if class_weights is None:
            self._loss = CrossEntropyLoss()
        else:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
            self._loss = CrossEntropyLoss(weight=class_weights_tensor)

    def _train_epoch(self, dataloader: DataLoader) -> None:
        """
        Trains the model for one epoch.

        :param dataloader: The DataLoader object to be used for training.
        :type dataloader: DataLoader
        """
        self._new_model.train()
        for batch_features, batch_targets in dataloader:
            self._optimizer.zero_grad()
            predictions = self._new_model(batch_features)
            predictions = predictions.view(-1, predictions.size(2))
            batch_targets = batch_targets.view(-1)

            loss = self._loss(predictions, batch_targets)
            loss.backward()
            self._optimizer.step()

    def _train_model(
        self,
        features: List[List[int]],
        targets: List[List[int]],
        epochs: int,
        batch_size: int,
        class_weights: List[float] = None,
    ) -> None:
        """
        Trains the model.

        :param features: The features to be used for training.
        :type features: List[List[int]]
        :param targets: The targets to be used for training.
        :type targets: List[List[int]]
        :param epochs: The number of epochs to train the model.
        :type epochs: int
        :param batch_size: The batch size to be used for training.
        :type batch_size: int
        :param class_weights: The class weights to be used for training.
        :type class_weights: List[float]
        """
        self._initialize_new_model()
        dataloader = self._prepare_data(features, targets, batch_size)
        self._set_loss_function(class_weights)

        for epoch in range(epochs):
            self._train_epoch(dataloader)

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
        """
        Trains the model asynchronously.

        :param features: The features to be used for training.
        :type features: List[List[int]]
        :param targets: The targets to be used for training.
        :type targets: List[List[int]]
        :param epochs: The number of epochs to train the model.
        :type epochs: int
        :param batch_size: The batch size to be used for training.
        :type batch_size: int
        :param class_weights: The class weights to be used for training.
        :type class_weights: List[float]
        """
        self._training_queue.put((features, targets, epochs, batch_size, class_weights))

    def _worker(self) -> None:
        """
        Worker function for training the model asynchronously.

        This function is run in a separate thread.
        """
        while True:
            features, targets, epochs, batch_size, class_weights = self._training_queue.get()
            self._train_model(features, targets, epochs, batch_size, class_weights)
            self._training_queue.task_done()

    def predict_with_confidence(self, unlabeled_sentence: List[int]) -> Tuple[List[int], List[float]]:
        """
        Predicts the labels for an unlabeled sentence.

        :param unlabeled_sentence: The unlabeled sentence to predict.
        :type unlabeled_sentence: List[int]
        :return: A tuple containing the predicted labels and confidence scores.
        :rtype: Tuple[List[int], List[float]]
        """
        features = torch.tensor([unlabeled_sentence], dtype=torch.long)

        self._model.eval()

        with torch.no_grad():
            probabilities = self._model(features).cpu()
            max_values, predictions = torch.max(probabilities, dim=-1)
            confidence_scores = max_values

        return predictions[0].tolist(), confidence_scores[0].tolist()

    def save(self, path: str) -> None:
        """
        Saves the model to a file.

        :param path: The path to save the model.
        :type path: str
        :raises ValueError: If the model, optimizer, or loss function is not initialized.
        """
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
        """
        Loads the model weights from a file.

        :param file_path: The path to load the model weights.
        :type file_path: str
        :raises ValueError: If the model is not initialized.
        """
        if self._model is None:
            raise ValueError("Model must be initialized before loading weights.")

        state_dict = torch.load(file_path)
        self._model.load_state_dict(state_dict["model_state_dict"])
        self._initialize_optimizer(state_dict)
        self._initialize_loss(state_dict)

    def _initialize_optimizer(self, state_dict: dict) -> None:
        """
        Initializes the optimizer based on the state dictionary.

        :param state_dict: The state dictionary to initialize the optimizer.
        :type state_dict: dict
        """
        if "optimizer_state_dict" in state_dict and "optimizer_name" in state_dict:
            optimizer_class = getattr(optim, state_dict["optimizer_name"])
            self._optimizer = optimizer_class(self._model.parameters())
            self._optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        else:
            self._optimizer = optim.Adam(self._model.parameters())

    def _initialize_loss(self, state_dict: dict) -> None:
        """
        Initializes the loss function based on the state dictionary.

        :param state_dict: The state dictionary to initialize the loss function.
        :type state_dict: dict
        """
        if "loss_state_dict" in state_dict and "loss_name" in state_dict:
            loss_class = getattr(torch.nn, state_dict["loss_name"])
            self._loss = loss_class()
            if state_dict["loss_state_dict"] is not None:
                state_dict["loss_state_dict"].pop("weight", None)
                self._loss.load_state_dict(state_dict["loss_state_dict"])
        else:
            self._loss = CrossEntropyLoss()

    def validate_torch_model(self, num_words: int, num_classes: int) -> None:
        """
        Validates the PyTorch model.

        :param num_words: The number of words in the vocabulary.
        :type num_words: int
        :param num_classes: The number of classes in the model.
        :type num_classes: int
        :raises ValueError: If the embedding size or output size does not match the expected values.
        """
        layers = list(self._model.children())
        embedding_size = layers[0].num_embeddings
        if embedding_size != num_words:
            raise ValueError(f"Expected embedding size {num_words}, but got {embedding_size}.")

        out_features = layers[-1].out_features
        if out_features != num_classes:
            raise ValueError(f"Expected output size {num_classes}, but got {out_features}.")

    def evaluate_metrics(self, features: List[List[int]], targets: List[List[int]]) -> Dict[str, float | int]:
        """
        Evaluates the model metrics.

        :param features: The features to be used for evaluation.
        :type features: List[List[int]]
        :param targets: The targets to be used for evaluation.
        :type targets: List[List[int]]
        :return: A dictionary containing the evaluation metrics.
        :rtype: Dict[str, float | int]
        """
        features_tensor, targets_tensor = self._prepare_tensors(features, targets)
        predictions_flat, targets_flat = self._get_predictions(features_tensor, targets_tensor)
        return self._calculate_metrics(targets_flat, predictions_flat)

    def _prepare_tensors(self, features: List[List[int]], targets: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares the tensors for evaluation.

        :param features: The features to be used for evaluation.
        :type features: List[List[int]]
        :param targets: The targets to be used for evaluation.
        :type targets: List[List[int]]
        :return: A tuple containing the features and targets tensors.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        features_tensor = pad_sequence(
            [torch.tensor(f, dtype=torch.long) for f in features],
            batch_first=True,
        )
        targets_tensor = pad_sequence(
            [torch.tensor(t, dtype=torch.long) for t in targets],
            batch_first=True,
        )
        return features_tensor, targets_tensor

    def _get_predictions(self, features_tensor: torch.Tensor, targets_tensor: torch.Tensor):
        """
        Gets the predictions for the features tensor.

        :param features_tensor: The features tensor to get predictions.
        :type features_tensor: torch.Tensor
        :param targets_tensor: The targets tensor to get predictions.
        :type targets_tensor: torch.Tensor
        :return: A tuple containing the predictions and targets.
        :rtype: Tuple[List[int], List[int]]
        """
        self._model.eval()
        with torch.no_grad():
            probabilities = self._model(features_tensor).cpu()
            _, predictions = torch.max(probabilities, dim=-1)
        return predictions.view(-1).tolist(), targets_tensor.view(-1).tolist()

    def _calculate_metrics(self, targets_flat: List[int], predictions_flat: List[int]) -> Dict[str, float]:
        """
        Calculates the evaluation metrics.

        :param targets_flat: The flattened targets.
        :type targets_flat: List[int]
        :param predictions_flat: The flattened predictions.
        :type predictions_flat: List[int]
        :return: A dictionary containing the evaluation metrics.
        :rtype: Dict[str, float]
        """
        accuracy = accuracy_score(targets_flat, predictions_flat)
        precision = precision_score(targets_flat, predictions_flat, average="weighted", zero_division=0)
        recall = recall_score(targets_flat, predictions_flat, average="weighted", zero_division=0)
        f1 = f1_score(targets_flat, predictions_flat, average="weighted", zero_division=0)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
