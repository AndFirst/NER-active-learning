from app.data_types import DatasetConf, ProjectFormState
from app.constants import (
    DEFAULT_PADDING_IDX,
    DEFAULT_PADDING_LABEL,
    DEFAULT_UNLABELED_LABEL,
    DEFAULT_UNLABELED_IDX,
    DEFAULT_INPUT_EXTENSION,
    DEFAULT_OUTPUT_EXTENSION,
    DEFAULT_MAX_SENTENCE_LENGTH,
)


def test_from_state():
    project_form_state = ProjectFormState(
        save_path="/path/to/save",
        output_extension=".out",
        dataset_path="/path/to/dataset.csv",
    )

    dataset_conf = DatasetConf.from_state(project_form_state)

    assert dataset_conf.unlabeled_path == "/path/to/save/unlabeled.csv"
    assert dataset_conf.labeled_path == "/path/to/save/labeled.out"
    assert dataset_conf.labels_to_idx_path == "/path/to/save/labels_to_idx.json"
    assert dataset_conf.padding_label == DEFAULT_PADDING_LABEL
    assert dataset_conf.padding_idx == DEFAULT_PADDING_IDX
    assert dataset_conf.unlabeled_label == DEFAULT_UNLABELED_LABEL
    assert dataset_conf.unlabeled_idx == DEFAULT_UNLABELED_IDX
    assert dataset_conf.max_sentence_length == DEFAULT_MAX_SENTENCE_LENGTH


def test_from_state_with_defaults():
    project_form_state = ProjectFormState(
        save_path="/path/to/save",
    )

    dataset_conf = DatasetConf.from_state(project_form_state)

    assert dataset_conf.unlabeled_path == f"/path/to/save/unlabeled{DEFAULT_INPUT_EXTENSION}"
    assert dataset_conf.labeled_path == f"/path/to/save/labeled{DEFAULT_OUTPUT_EXTENSION}"
    assert dataset_conf.labels_to_idx_path == "/path/to/save/labels_to_idx.json"
    assert dataset_conf.padding_label == DEFAULT_PADDING_LABEL
    assert dataset_conf.padding_idx == DEFAULT_PADDING_IDX
    assert dataset_conf.unlabeled_label == DEFAULT_UNLABELED_LABEL
    assert dataset_conf.unlabeled_idx == DEFAULT_UNLABELED_IDX
    assert dataset_conf.max_sentence_length == DEFAULT_MAX_SENTENCE_LENGTH


def test_from_dict():
    data = {
        "unlabeled_path": "/path/to/unlabeled.txt",
        "labeled_path": "/path/to/labeled.txt",
        "labels_to_idx_path": "/path/to/labels_to_idx.json",
        "padding_label": "PAD",
        "padding_idx": 0,
        "unlabeled_label": "UNLABELED",
        "unlabeled_idx": 1,
        "max_sentence_length": 50,
    }

    dataset_conf = DatasetConf.from_dict(data)

    assert dataset_conf.unlabeled_path == data["unlabeled_path"]
    assert dataset_conf.labeled_path == data["labeled_path"]
    assert dataset_conf.labels_to_idx_path == data["labels_to_idx_path"]
    assert dataset_conf.padding_label == data["padding_label"]
    assert dataset_conf.padding_idx == data["padding_idx"]
    assert dataset_conf.unlabeled_label == data["unlabeled_label"]
    assert dataset_conf.unlabeled_idx == data["unlabeled_idx"]
    assert dataset_conf.max_sentence_length == data["max_sentence_length"]


def test_to_dict():
    dataset_conf = DatasetConf(
        unlabeled_path="/path/to/unlabeled.txt",
        labeled_path="/path/to/labeled.txt",
        labels_to_idx_path="/path/to/labels_to_idx.json",
        padding_label="PAD",
        padding_idx=0,
        unlabeled_label="UNLABELED",
        unlabeled_idx=1,
        max_sentence_length=50,
    )

    expected_dict = {
        "unlabeled_path": "/path/to/unlabeled.txt",
        "labeled_path": "/path/to/labeled.txt",
        "labels_to_idx_path": "/path/to/labels_to_idx.json",
        "padding_label": "PAD",
        "padding_idx": 0,
        "unlabeled_label": "UNLABELED",
        "unlabeled_idx": 1,
        "max_sentence_length": 50,
    }

    assert dataset_conf.to_dict() == expected_dict


def test_get_existing_property():
    dataset_conf = DatasetConf(
        unlabeled_path="/path/to/unlabeled.txt",
        labeled_path="/path/to/labeled.txt",
        labels_to_idx_path="/path/to/labels_to_idx.json",
        padding_label="PAD",
        padding_idx=0,
        unlabeled_label="UNLABELED",
        unlabeled_idx=1,
        max_sentence_length=50,
    )

    assert dataset_conf.get("unlabeled_path", "default") == "/path/to/unlabeled.txt"
    assert dataset_conf.get("nonexistent_property", "default") == "default"


def test_get_default_property():
    dataset_conf = DatasetConf(
        unlabeled_path="/path/to/unlabeled.txt",
        labeled_path="/path/to/labeled.txt",
        labels_to_idx_path="/path/to/labels_to_idx.json",
        padding_label="PAD",
        padding_idx=0,
        unlabeled_label="UNLABELED",
        unlabeled_idx=1,
        max_sentence_length=50,
    )

    assert dataset_conf.get("nonexistent_property", "default") == "default"


def test_default_values():
    dataset_conf = DatasetConf(
        unlabeled_path="/path/to/unlabeled.txt",
        labeled_path="/path/to/labeled.txt",
        labels_to_idx_path="/path/to/labels_to_idx.json",
        padding_label=DEFAULT_PADDING_LABEL,
        padding_idx=DEFAULT_PADDING_IDX,
        unlabeled_label=DEFAULT_UNLABELED_LABEL,
        unlabeled_idx=DEFAULT_UNLABELED_IDX,
        max_sentence_length=DEFAULT_MAX_SENTENCE_LENGTH,
    )

    assert dataset_conf.padding_label == DEFAULT_PADDING_LABEL
    assert dataset_conf.padding_idx == DEFAULT_PADDING_IDX
    assert dataset_conf.unlabeled_label == DEFAULT_UNLABELED_LABEL
    assert dataset_conf.unlabeled_idx == DEFAULT_UNLABELED_IDX
    assert dataset_conf.max_sentence_length == DEFAULT_MAX_SENTENCE_LENGTH
