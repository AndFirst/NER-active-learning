from __future__ import annotations

import itertools
from dataclasses import dataclass, field
import os
from typing import Tuple, Optional, Any, List, Dict
from app.data_types import (
    ProjectFormState,
    DatasetConf,
    AssistantConf,
    ModelConf,
    ProjectConf)
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
    def create(cls, project_form_state: ProjectFormState) -> None:
        os.makedirs(project_form_state.save_path)

        # Create Assistant Config object
        assistant_conf = AssistantConf.create_from_state(project_form_state)

        # Create Dataset Config object
        dataset_conf = DatasetConf.create_from_state(project_form_state)

        # Copy dataset to project's path
        shutil.copy(
            project_form_state.dataset_path, dataset_conf.unlabeled_path
        )

        # Save word to indexes
        unlabeled_file = Factory.create_unlabeled_file(dataset_conf.unlabeled_path)
        word_to_idx = Project.create_word_to_idx(unlabeled_file.unique_words())
        with open(dataset_conf.words_to_idx_path, "w") as word_to_idx_file:
            json.dump(word_to_idx, word_to_idx_file)

        # Save label to indexes
        label_to_idx = Project.create_label_to_idx(AssistantConf.get_labelset())
        with open(dataset_conf.labels_to_idx_path, "w") as label_to_idx_file:
            json.dump(label_to_idx, label_to_idx_file)

        # Create Model Config object
        model_conf = ModelConf.create_from_state(project_form_state,
                                                 unlabeled_file.unique_words(),
                                                 len(AssistantConf.get_labelset)*2 + 1)

        # Copy implementation of model if it is custom
        if model_conf.is_custom_model_type():
            src_model_implementation = project_form_state.model_implementation_path
            shutil.copy(
                src_model_implementation,
                model_conf.implementation_path,
            )

        # Copy model state if it exists
        if project_form_state.model_state_path:
            shutil.copy(project_form_state.model_state_path, model_conf.model_state_path)

        # Create and save model
        model = Factory.create_model(model_conf)
        model.save(model_conf.model_state_path)

        # Create Project Config object
        project_conf = ProjectConf.create_from_state(project_form_state,
                                                     model_conf,
                                                     assistant_conf,
                                                     dataset_conf)

        # save config
        project_conf.save_config(project_form_state.save_path)

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
