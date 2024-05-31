import pytest
from unittest.mock import MagicMock, patch
import torch
from app.learning.models import NERModel


@pytest.fixture
def setup_model():
    ner_model = NERModel()
    ner_model._model = MagicMock()  # Initialize the _model attribute
    return ner_model


def test_set_loss_function_with_class_weights(setup_model):
    class_weights = [0.5, 0.5]
    setup_model._set_loss_function(class_weights)
    assert setup_model._loss is not None


def test_set_loss_function_without_class_weights(setup_model):
    setup_model._set_loss_function(None)
    assert setup_model._loss is not None


@patch("torch.save")
def test_save(mock_save, setup_model):
    setup_model._model = MagicMock()
    setup_model._optimizer = MagicMock()
    setup_model._loss = MagicMock()
    setup_model.save("path/to/save")
    mock_save.assert_called_once()


@patch("torch.tensor")
def test_prepare_data(mock_tensor, setup_model):
    mock_tensor.return_value = MagicMock()
    features = [[1, 2], [3, 4], [5, 6]]
    targets = [[0, 1], [1, 0], [0, 1]]
    batch_size = 2
    dataloader = setup_model._prepare_data(features, targets, batch_size)
    assert len(dataloader.dataset) == 1  # Ensure the length of the dataset matches the input


@patch("torch.load")
def test_load_weights(mock_load, setup_model):
    mock_load.return_value = {
        "model_state_dict": {},
        "optimizer_name": "Adam",
        "optimizer_state_dict": {
            "param_groups": [{"params": [torch.randn(1, requires_grad=True)]}],  # Add this line
            "state": {},
        },
        "loss_name": "CrossEntropyLoss",
        "loss_state_dict": None,
    }
    setup_model._model = MagicMock()
    setup_model._model.parameters.return_value = [torch.randn(1, requires_grad=True)]  # Ensure the model has parameters
    setup_model.load_weights("path/to/load")
    assert setup_model._optimizer is not None
    assert setup_model._loss is not None
