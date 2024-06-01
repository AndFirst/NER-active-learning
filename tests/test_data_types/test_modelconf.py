import os

from app.data_types import ProjectFormState, ModelConf, LabelData
from app.constants import (
    DEFAULT_DROPOUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_WORDS,
)


def test_from_state():
    project_form_state = ProjectFormState(
        name="TestProject",
        save_path="/path/to/save",
        model_type="custom",
        description="A test project",
        dataset_path="/path/to/dataset.csv",
        labels=[LabelData("label1", (255, 0, 0, 255))],
        model_state_path="model.pth",
        model_implementation_path="/path/to/model/implementation",
    )

    model_conf = ModelConf.from_state(project_form_state, 10)

    assert model_conf.type == "custom"
    assert model_conf.state_path == "/path/to/save/" + "model.pth"
    assert model_conf.dropout == DEFAULT_DROPOUT
    assert model_conf.learning_rate == DEFAULT_LEARNING_RATE
    assert model_conf.num_words == DEFAULT_NUM_WORDS
    assert model_conf.num_labels == 10
    assert model_conf.num_classes == 21
    assert model_conf.implementation_path == "app/learning/models/custom_model_TestProject.py"


def test_from_dict():
    directory_path = "/path/to"
    data = {
        "type": "custom",
        "state_path": "model.pth",
        "dropout": 0.5,
        "learning_rate": 0.001,
        "num_words": 1000,
        "num_labels": 10,
        "num_classes": 21,
        "implementation_path": "app/learning/models/custom_model_TestProject.py",
    }

    model_conf = ModelConf.from_dict(directory_path, data)

    assert model_conf.type == data["type"]
    assert model_conf.state_path == os.path.join(directory_path, data["state_path"])
    assert model_conf.dropout == data["dropout"]
    assert model_conf.learning_rate == data["learning_rate"]
    assert model_conf.num_words == data["num_words"]
    assert model_conf.num_labels == data["num_labels"]
    assert model_conf.num_classes == data["num_classes"]
    assert model_conf.implementation_path == data["implementation_path"]


def test_to_dict():
    model_conf = ModelConf(
        type="custom",
        state_path="/path/to/model.pth",
        dropout=0.5,
        learning_rate=0.001,
        num_words=1000,
        num_labels=10,
        num_classes=21,
        implementation_path="app/learning/models/custom_model_TestProject.py",
    )

    expected_dict = {
        "type": "custom",
        "state_path": "model.pth",
        "dropout": 0.5,
        "learning_rate": 0.001,
        "num_words": 1000,
        "num_labels": 10,
        "num_classes": 21,
        "implementation_path": "app/learning/models/custom_model_TestProject.py",
    }

    assert model_conf.to_dict() == expected_dict


def test_is_custom_model_type():
    model_conf = ModelConf(
        type="custom",
        state_path="/path/to/model.pth",
        dropout=0.5,
        learning_rate=0.001,
        num_words=1000,
        num_labels=10,
        num_classes=21,
        implementation_path="app/learning/models/custom_model_TestProject.py",
    )

    assert model_conf.is_custom_model_type() is True

    model_conf.type = "standard"
    assert model_conf.is_custom_model_type() is False


def test_get_existing_property():
    model_conf = ModelConf(
        type="custom",
        state_path="/path/to/model.pth",
        dropout=0.5,
        learning_rate=0.001,
        num_words=1000,
        num_labels=10,
        num_classes=21,
        implementation_path="app/learning/models/custom_model_TestProject.py",
    )

    assert model_conf.get("type", "default") == "custom"
    assert model_conf.get("nonexistent_property", "default") == "default"


def test_get_default_property():
    model_conf = ModelConf(
        type="custom",
        state_path="/path/to/model.pth",
        dropout=0.5,
        learning_rate=0.001,
        num_words=1000,
        num_labels=10,
        num_classes=21,
        implementation_path="app/learning/models/custom_model_TestProject.py",
    )

    assert model_conf.get("nonexistent_property", "default") == "default"
