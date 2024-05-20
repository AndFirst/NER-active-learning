from __future__ import annotations

import itertools
import os

import json
import shutil
from typing import Dict, Set

from app.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_DROPOUT,
    DEFAULT_INPUT_EXTENSION,
    DEFAULT_OUTPUT_EXTENSION,
    DEFAULT_PADDING_LABEL,
    DEFAULT_PADDING_IDX,
    DEFAULT_UNLABELED_LABEL,
    DEFAULT_UNLABELED_IDX,
    DEFAULT_LEARNING_RATE,
)
from app.learning.active_learning import ActiveLearningManager
from app.learning.dataset.dataset import Dataset
from app.learning.factory import Factory
from app.learning.models.ner_model import NERModel


class Project:
    def __init__(
        self,
        assistant: ActiveLearningManager,
        dataset: Dataset,
        model: NERModel,
    ):
        self._assistant = assistant
        self._dataset = dataset
        self._model = model

    @classmethod
    def load(cls, directory_path: str) -> Project:
        config_file = f"{directory_path}/project.json"
        if not os.path.isfile(config_file):
            raise FileNotFoundError("Project configuration file not found.")

        with open(config_file, "r") as config_file:
            config = json.load(config_file)

        dataset_conf = config["dataset"]
        dataset = Factory.create_dataset(dataset_conf)

        model_conf = config["model"]
        model = Factory.create_model(model_conf)

        assistant_conf = config["assistant"]
        assistant = Factory.create_assistant(model, dataset, assistant_conf)

        return Project(assistant, dataset, model)

    def save(self, directory_path: str) -> None:
        config_file = f"{directory_path}/project.json"
        if not os.path.isfile(config_file):
            raise FileNotFoundError("Project configuration file not found.")

        with open(config_file, "r") as config_file:
            config = json.load(config_file)
            model_path = config["model"]["model_state_path"]

        self._model.save(model_path)
        self._dataset.save()

    @classmethod
    def create(cls, directory_path: str, project_state: dict) -> None:
        os.makedirs(directory_path)

        assistant_conf = {
            "batch_size": project_state.get("batch_size", DEFAULT_BATCH_SIZE),
            "epochs": project_state.get("epochs", DEFAULT_EPOCHS),
            "labels": {
                label["label"]: label["color"]
                for label in project_state["labels"]
            },
        }

        input_extension = project_state.get(
            "input_extension", DEFAULT_INPUT_EXTENSION
        )
        output_extension = project_state.get(
            "output_extension", DEFAULT_OUTPUT_EXTENSION
        )

        dataset_conf = {
            "unlabeled_path": f"{directory_path}/unlabeled{input_extension}",
            "labeled_path": f"{directory_path}/labeled{output_extension}",
            "words_to_idx_path": f"{directory_path}/words_to_idx.json",
            "labels_to_idx_path": f"{directory_path}/labels_to_idx.json",
            "padding_label": DEFAULT_PADDING_LABEL,
            "padding_idx": DEFAULT_PADDING_IDX,
            "unlabeled_label": DEFAULT_UNLABELED_LABEL,
            "unlabeled_idx": DEFAULT_UNLABELED_IDX,
        }
        # copy dataset to our directory
        shutil.copy(
            project_state["dataset_path"], dataset_conf["unlabeled_path"]
        )

        with open(dataset_conf["labeled_path"], "w"):
            pass

        unlabeled_file = Factory.create_unlabeled_file(
            dataset_conf["unlabeled_path"]
        )
        labeled_file = Factory.create_labeled_file(
            dataset_conf["labeled_path"]
        )

        unique_words = unlabeled_file.unique_words()

        word_to_idx = Project.create_word_to_idx(unique_words)
        with open(dataset_conf["words_to_idx_path"], "w") as word_to_idx_file:
            json.dump(word_to_idx, word_to_idx_file)

        labels = {label["label"] for label in project_state["labels"]}
        label_to_idx = Project.create_label_to_idx(labels)

        with open(
            dataset_conf["labels_to_idx_path"], "w"
        ) as label_to_idx_file:
            json.dump(label_to_idx, label_to_idx_file)

        model_conf = {
            "model_type": project_state.get("model_type"),
            "model_state_path": f"{directory_path}/model.pth",
            "dropout": project_state.get("dropout", DEFAULT_DROPOUT),
            "learning_rate": project_state.get(
                "learning_rate", DEFAULT_LEARNING_RATE
            ),
            "num_words": len(unique_words),
            "num_classes": len(labels) * 2 + 1,
        }
        print(model_conf)
        if model_conf["model_type"] == "custom":
            model_conf["model_implementation_path"] = (
                f"app/learning/models/custom_model_{project_state['name']}.py"
            )
            source_model_implementation = project_state.get(
                "model_implementation_path"
            )
            shutil.copy(
                source_model_implementation,
                model_conf["model_implementation_path"],
            )

        source_model_state = project_state.get("model_state_path")
        if source_model_state:
            shutil.copy(source_model_state, model_conf["model_state_path"])

        project_conf = {
            "name": project_state["name"],
            "description": project_state["description"],
            "model": model_conf,
            "assistant": assistant_conf,
            "dataset": dataset_conf,
        }

        # save config
        with open(
            f"{directory_path}/project.json", "w"
        ) as project_config_file:
            json.dump(project_conf, project_config_file)

        model = Factory.create_model(model_conf)
        model.save(model_conf["model_state_path"])

    @staticmethod
    def create_word_to_idx(words: Set[str]) -> Dict[str, int]:
        word_to_idx = {DEFAULT_PADDING_LABEL: DEFAULT_PADDING_IDX}
        for idx, word in enumerate(words, 1):
            word_to_idx[word] = idx
        return word_to_idx

    @staticmethod
    def create_label_to_idx(labels: Set[str]) -> Dict[str, int]:
        label_to_idx = {DEFAULT_UNLABELED_LABEL: DEFAULT_UNLABELED_IDX}
        prefixes = "B-", "I-"
        labels_with_prefixes = itertools.product(prefixes, labels)
        for idx, pair in enumerate(labels_with_prefixes, 1):
            label = "".join(pair)
            label_to_idx[label] = idx
        return label_to_idx

    def get_assistant(self) -> ActiveLearningManager:
        return self._assistant

    def get_model(self) -> NERModel:
        return self._model

    def get_labels(self) -> dict:
        return self._assistant.label_mapping
