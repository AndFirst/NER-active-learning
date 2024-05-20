from importlib.util import spec_from_file_location, module_from_spec
import os
import sys
import threading

import torch

from app.learning.models.ner_model import NERModel

import inspect
import torch.nn as nn


class CustomModel(NERModel):
    def __init__(self, model_implementation_path: str) -> None:
        super().__init__()
        self._model = None
        self._new_model = None
        self._optimizer = None
        self._loss = None
        self._lock = threading.Lock()

        # Add the directory containing the model implementation to the Python path
        sys.path.append(os.path.dirname(model_implementation_path))

        # Import the module containing the model
        spec = spec_from_file_location(
            "model_module", model_implementation_path
        )
        model_module = module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # Get all classes in the module that inherit from torch.nn.Module
        model_classes = [
            obj
            for name, obj in inspect.getmembers(model_module)
            if inspect.isclass(obj) and issubclass(obj, nn.Module)
        ]

        # If there are multiple or no such classes, raise an error
        if len(model_classes) > 1:
            raise ValueError(
                "The module contains multiple classes that inherit from torch.nn.Module. Please ensure there is only one such class."
            )
        elif not model_classes:
            raise ValueError(
                "The module does not contain a class that inherits from torch.nn.Module."
            )

        # Get the model class from the module
        self._model = model_classes[0]()

    def save(self, path: str) -> None:
        if self._model is None:
            raise ValueError("Model has not been instantiated yet.")
        torch.save(self._model.state_dict(), path)

    def load_weights(self, path: str) -> None:
        if self._model is None:
            raise ValueError("Model has not been instantiated yet.")
        state_dict = torch.load(path)
        self._model.load_state_dict(state_dict)
        self._model.eval()
        print(self._model)
