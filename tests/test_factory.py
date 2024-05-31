from unittest.mock import MagicMock, patch, mock_open

import pytest

from app.learning.factory import Factory
from app.data_types import DatasetConf, ModelConf, AssistantConf, LabelData


@patch("app.learning.factory.CsvTabSeparatedStrategy")
@patch("app.learning.factory.JsonListStrategy")
def test_create_data_persistence_strategy(mock_json_strategy, mock_csv_strategy):
    csv_path = "test.csv"
    json_path = "test.json"
    Factory.create_data_persistence_strategy(csv_path)
    mock_csv_strategy.assert_called_once_with(csv_path)
    Factory.create_data_persistence_strategy(json_path)
    mock_json_strategy.assert_called_once_with(json_path)


@patch("app.learning.factory.LabeledSentenceRepository")
def test_create_labeled_repository(mock_repo):
    path = "test.csv"
    Factory.create_labeled_repository(path)
    mock_repo.assert_called_once()


@patch("app.learning.factory.UnlabeledSentenceRepository")
def test_create_unlabeled_repository(mock_repo):
    path = "test.csv"
    Factory.create_unlabeled_repository(path)
    mock_repo.assert_called_once()


def test_create_data_persistence_strategy_with_unsupported_extension():
    unsupported_file_path = "test.unsupported"
    with pytest.raises(ValueError, match="Unsupported extension: .unsupported"):
        Factory.create_data_persistence_strategy(unsupported_file_path)


@patch("app.learning.factory.ActiveLearningManager")
def test_create_assistant(mock_manager):
    model = MagicMock()
    dataset = MagicMock()
    config = AssistantConf(
        batch_size=16,
        sampling_batch_size=16,
        epochs=20,
        labels=[LabelData(label="Label1", color=(255, 0, 0, 255))],
    )
    Factory.create_assistant(model, dataset, config)
    mock_manager.assert_called_once_with(model, dataset, config)


@patch("app.learning.factory.BiLSTMClassifier")
def test_create_model_bilstm(mock_bilstm):
    common_params = {
        "state_path": "test.pth",
        "num_words": 1000,
        "num_classes": 5,
        "learning_rate": 0.001,
        "num_labels": 11,
        "dropout": 0.5,
    }
    model_params = {
        "num_words": 1000,
        "num_classes": 5,
        "learning_rate": 0.001,
    }
    bilstm_config = ModelConf(type="BiLSTM", **common_params)
    Factory.create_model(bilstm_config)
    mock_bilstm.assert_called_once_with(**model_params)


@patch("app.learning.factory.CustomModel")
def test_create_model_custom(mock_custom):
    common_params = {
        "state_path": "test.pth",
        "num_words": 1000,
        "num_classes": 5,
        "learning_rate": 0.001,
        "num_labels": 11,
        "dropout": 0.5,
        "implementation_path": "test_model.py",
    }
    custom_config = ModelConf(type="custom", **common_params)
    Factory.create_model(custom_config)
    mock_custom.assert_called_once_with(common_params["implementation_path"])


def test_create_model_with_unsupported_type():
    common_params = {
        "state_path": "test.pth",
        "num_words": 1000,
        "num_classes": 5,
        "learning_rate": 0.001,
        "num_labels": 11,
        "dropout": 0.5,
    }
    unsupported_config = ModelConf(type="unsupported", **common_params)
    with pytest.raises(ValueError, match="Unsupported model type: unsupported"):
        Factory.create_model(unsupported_config)


@patch("app.learning.factory.Dataset")
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"labels": {"label1": 0, "label2": 1}}',
)
def test_create_dataset(mock_open, mock_dataset):
    cfg = DatasetConf(
        labels_to_idx_path="labels.json",
        unlabeled_path="unlabeled.csv",
        labeled_path="labeled.csv",
        padding_idx=0,
        padding_label="",
        unlabeled_idx=0,
        unlabeled_label="",
        max_sentence_length=100,
    )
    Factory.create_dataset(cfg)
    mock_dataset.assert_called_once()


@patch("os.path.exists")
@patch("app.learning.factory.BiLSTMClassifier")
def test_load_weights_bilstm(mock_bilstm, mock_exists):
    mock_exists.return_value = True
    mock_bilstm_instance = mock_bilstm.return_value
    common_params = {
        "state_path": "test.pth",
        "num_words": 1000,
        "num_classes": 5,
        "learning_rate": 0.001,
        "dropout": 0.1,
        "num_labels": 2,
    }
    bilstm_config = ModelConf(type="BiLSTM", **common_params)
    Factory.create_model(bilstm_config)
    mock_bilstm_instance.load_weights.assert_called_once_with(common_params["state_path"])


@patch("os.path.exists")
@patch("app.learning.factory.CustomModel")
def test_load_weights_custom(mock_custom, mock_exists):
    mock_exists.return_value = True
    mock_custom_instance = mock_custom.return_value
    common_params = {
        "state_path": "test.pth",
        "num_words": 1000,
        "num_classes": 5,
        "learning_rate": 0.001,
        "implementation_path": "test_model.py",
        "dropout": 0.1,
        "num_labels": 2,
    }
    custom_config = ModelConf(type="custom", **common_params)
    Factory.create_model(custom_config)
    mock_custom_instance.load_weights.assert_called_once_with(common_params["state_path"])
