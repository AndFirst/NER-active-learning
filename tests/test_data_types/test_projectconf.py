import json
from unittest.mock import mock_open, patch

import pytest

from app.data_types import (
    ProjectConf,
    ProjectFormState,
    ModelConf,
    AssistantConf,
    DatasetConf,
    LabelData,
)

DEFAULT_DROPOUT = 0.5
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_INPUT_EXTENSION = ".csv"
DEFAULT_OUTPUT_EXTENSION = ".out"
DEFAULT_PADDING_LABEL = "PAD"
DEFAULT_PADDING_IDX = 0
DEFAULT_UNLABELED_LABEL = "UNLABELED"
DEFAULT_UNLABELED_IDX = 1
DEFAULT_SAMPLING_BATCH_SIZE = 4
MAX_SENTENCE_LENGTH = 50

model_conf = ModelConf(
    type="custom",
    state_path="/path/to/model.pth",
    dropout=DEFAULT_DROPOUT,
    learning_rate=DEFAULT_LEARNING_RATE,
    num_words=1000,
    num_labels=10,
    num_classes=21,
    implementation_path="app/learning/models/custom_model_TestProject.py",
)

assistant_conf = AssistantConf(
    batch_size=DEFAULT_BATCH_SIZE,
    sampling_batch_size=DEFAULT_SAMPLING_BATCH_SIZE,
    epochs=DEFAULT_EPOCHS,
    labels=[LabelData(label="Label1", color=(255, 0, 0, 255))],
)

dataset_conf = DatasetConf(
    unlabeled_path="/path/to/unlabeled.csv",
    labeled_path="/path/to/labeled.out",
    words_to_idx_path="/path/to/words_to_idx.json",
    labels_to_idx_path="/path/to/labels_to_idx.json",
    padding_label=DEFAULT_PADDING_LABEL,
    padding_idx=DEFAULT_PADDING_IDX,
    unlabeled_label=DEFAULT_UNLABELED_LABEL,
    unlabeled_idx=DEFAULT_UNLABELED_IDX,
    max_sentence_length=MAX_SENTENCE_LENGTH,
)

project_form_state = ProjectFormState(
    name="TestProject",
    description="A test project",
    save_path="/path/to/save",
    model_type="custom",
    dataset_path="/path/to/dataset.csv",
)


def test_from_state():
    project_conf = ProjectConf.from_state(
        project_form_state, model_conf, assistant_conf, dataset_conf
    )

    assert project_conf.name == "TestProject"
    assert project_conf.description == "A test project"
    assert project_conf.model_conf == model_conf
    assert project_conf.assistant_conf == assistant_conf
    assert project_conf.dataset_conf == dataset_conf


def test_to_dict():
    project_conf = ProjectConf(
        name="TestProject",
        description="A test project",
        model_conf=model_conf,
        assistant_conf=assistant_conf,
        dataset_conf=dataset_conf,
    )

    expected_dict = {
        "name": "TestProject",
        "description": "A test project",
        "model": model_conf.to_dict(),
        "assistant": assistant_conf.to_dict(),
        "dataset": dataset_conf.to_dict(),
    }

    assert project_conf.to_dict() == expected_dict


@patch("builtins.open", new_callable=mock_open)
def test_save_config(mock_open):
    project_conf = ProjectConf(
        name="TestProject",
        description="A test project",
        model_conf=model_conf,
        assistant_conf=assistant_conf,
        dataset_conf=dataset_conf,
    )

    with patch("json.dump") as mock_json_dump:
        project_conf.save_config("/path/to/save")

        mock_open.assert_called_once_with("/path/to/save/project.json", "w")
        mock_json_dump.assert_called_once_with(
            project_conf.to_dict(), mock_open()
        )


def test_from_dict():
    data = {
        "name": "TestProject",
        "description": "A test project",
        "model": model_conf.to_dict(),
        "assistant": assistant_conf.to_dict(),
        "dataset": dataset_conf.to_dict(),
    }

    project_conf = ProjectConf.from_dict(data)

    assert project_conf.name == data["name"]
    assert project_conf.description == data["description"]
    assert project_conf.model_conf == model_conf
    assert project_conf.assistant_conf == assistant_conf
    assert project_conf.dataset_conf == dataset_conf


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data=json.dumps(
        {
            "name": "TestProject",
            "description": "A test project",
            "model": model_conf.to_dict(),
            "assistant": assistant_conf.to_dict(),
            "dataset": dataset_conf.to_dict(),
        }
    ),
)
def test_from_file(mock_open):
    path = "/path/to/project.json"

    with patch("os.path.isfile", return_value=True):
        project_conf = ProjectConf.from_file(path)

    mock_open.assert_called_once_with(path, "r")
    assert project_conf.name == "TestProject"
    assert project_conf.description == "A test project"
    assert project_conf.model_conf == model_conf
    assert project_conf.assistant_conf == assistant_conf
    assert project_conf.dataset_conf == dataset_conf


def test_from_file_nonexistent_file():
    path = "/path/to/project.json"

    with pytest.raises(FileNotFoundError):
        with patch("os.path.isfile", return_value=False):
            ProjectConf.from_file(path)


def test_get_existing_property():
    project_conf = ProjectConf(
        name="TestProject",
        description="A test project",
        model_conf=model_conf,
        assistant_conf=assistant_conf,
        dataset_conf=dataset_conf,
    )

    assert project_conf.get("name", "default") == "TestProject"
    assert project_conf.get("nonexistent_property", "default") == "default"


def test_get_default_property():
    project_conf = ProjectConf(
        name="TestProject",
        description="A test project",
        model_conf=model_conf,
        assistant_conf=assistant_conf,
        dataset_conf=dataset_conf,
    )

    assert project_conf.get("nonexistent_property", "default") == "default"
