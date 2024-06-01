import threading
from importlib.util import spec_from_file_location, module_from_spec
import os
import sys
from queue import Queue

import torch

from app.learning.models.base import NERModel

import inspect
import torch.nn as nn


class CustomModel(NERModel):
    def __init__(self, model_implementation_path: str) -> None:
        super(CustomModel, self).__init__()

        # Add the directory containing the model implementation to the Python
        # path
        sys.path.append(os.path.dirname(model_implementation_path))

        # Import the module containing the model
        spec = spec_from_file_location("model_module", model_implementation_path)
        model_module = module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # Get all classes in the module that inherit from torch.nn.Module
        model_classes = [
            obj for name, obj in inspect.getmembers(model_module) if inspect.isclass(obj) and issubclass(obj, nn.Module)
        ]

        # If there are multiple or no such classes, raise an error
        if len(model_classes) > 1:
            raise ValueError(
                "The module contains multiple classes that inherit from torch.nn.Module.\
                Please ensure there is only one such class."
            )
        elif not model_classes:
            raise ValueError("The module does not contain a class that inherits from torch.nn.Module.")

        # Get the model class from the module
        self._model = model_classes[0]()
        self._loss = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        self._new_model = None
        self._lock = threading.Lock()
        self._training_queue = Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
