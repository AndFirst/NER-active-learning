from app.data_types import ProjectFormState, LabelData, AssistantConf
from app.constants import DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE
from app.exceptions import NoLabelFoundError
import pytest


def test_from_state():
    label_data = [LabelData(label="Label1", color=(255, 0, 0, 255))]
    assistant_conf = AssistantConf(
        batch_size="32",
        epochs="10",
        labels=label_data,
    )

    assert assistant_conf.batch_size == "32"
    assert assistant_conf.epochs == "10"
    assert assistant_conf.labels == label_data


def test_from_state_with_defaults():
    project_form_state = ProjectFormState()

    assistant_conf = AssistantConf.from_state(project_form_state)

    assert assistant_conf.batch_size == DEFAULT_BATCH_SIZE
    assert assistant_conf.epochs == DEFAULT_EPOCHS
    assert assistant_conf.labels == project_form_state.labels


def test_from_dict():
    data = {
        "batch_size": "32",
        "epochs": "10",
        "labels": [{"label": "Label1", "color": (255, 0, 0, 255)}],
    }

    assistant_conf = AssistantConf.from_dict(data)

    assert assistant_conf.batch_size == data["batch_size"]
    assert assistant_conf.epochs == data["epochs"]
    assert assistant_conf.labels == [LabelData(label="Label1", color=(255, 0, 0, 255))]


def test_to_dict():
    label1 = LabelData(label="Label1", color=(255, 0, 0, 255))
    assistant_conf = AssistantConf(batch_size="32", epochs="10", labels=[label1])

    expected_dict = {
        "batch_size": "32",
        "epochs": "10",
        "labels": [label1.to_dict()],
    }

    assert assistant_conf.to_dict() == expected_dict


def test_get_label():
    label1 = LabelData(label="Label1", color=(255, 0, 0, 255))
    assistant_conf = AssistantConf(batch_size="32", epochs="10", labels=[label1])

    assert assistant_conf.get_label("Label1") == label1

    with pytest.raises(NoLabelFoundError):
        assistant_conf.get_label("NonexistentLabel")


def test_get_labelset():
    label1 = LabelData(label="Label1", color=(255, 0, 0, 255))
    label2 = LabelData(label="Label2", color=(0, 255, 0, 255))
    assistant_conf = AssistantConf(
        batch_size="32", epochs="10", labels=[label1, label2]
    )

    expected_labelset = {"Label1", "Label2"}
    assert assistant_conf.get_labelset() == expected_labelset


def test_get_existing_property():
    assistant_conf = AssistantConf(
        batch_size="32",
        epochs="10",
        labels=[LabelData(label="Label1", color=(255, 0, 0, 255))],
    )

    assert assistant_conf.get("batch_size", "default") == "32"
    assert assistant_conf.get("nonexistent_property", "default") == "default"


def test_get_default_property():
    assistant_conf = AssistantConf(
        batch_size="32",
        epochs="10",
        labels=[LabelData(label="Label1", color=(255, 0, 0, 255))],
    )

    assert assistant_conf.get("nonexistent_property", "default") == "default"
