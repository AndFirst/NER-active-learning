from __future__ import annotations

import itertools
import os
from typing import Dict
from app.data_types import (
    ProjectFormState,
    DatasetConf,
    AssistantConf,
    ModelConf,
    ProjectConf,
    LabelData,
)
import json
import shutil
from typing import Set
from app.constants import (
    DEFAULT_PADDING_LABEL,
    DEFAULT_PADDING_IDX,
    DEFAULT_UNLABELED_LABEL,
    DEFAULT_UNLABELED_IDX,
    DEFAULT_NUM_WORDS,
)
from app.learning.active_learning import ActiveLearningManager
from app.learning.factory import Factory
from app.learning.models.base import NERModel


class Project:
    def __init__(self, config: ProjectConf, dir: str):
        self.config = config
        self.dir = dir
        self._dataset = Factory.create_dataset(config.dataset_conf)
        self._model = Factory.create_model(config.model_conf)
        self._assistant = Factory.create_assistant(
            self._model, self._dataset, config.assistant_conf
        )

    @classmethod
    def load(cls, dir_path: str) -> Project:
        config = ProjectConf.from_file(f"{dir_path}/project.json")
        return Project(config, dir_path)

    def save(self) -> None:
        self.config.save_config(self.dir)
        self._model.save(self.config.model_conf.state_path)
        self._dataset.save()

    @classmethod
    def create(cls, project_form_state: ProjectFormState) -> None:
        os.makedirs(project_form_state.save_path)

        # Create Assistant Config object
        assistant_conf = AssistantConf.from_state(project_form_state)

        # Create Dataset Config object
        dataset_conf = DatasetConf.from_state(project_form_state)

        # Copy dataset to project's path
        shutil.copy(
            project_form_state.dataset_path, dataset_conf.unlabeled_path
        )

        # Save word to indexes
        unlabeled_file = Factory.create_unlabeled_repository(
            dataset_conf.unlabeled_path
        )
        word_to_idx = Project.create_word_to_idx(unlabeled_file.unique_words())
        with open(dataset_conf.words_to_idx_path, "w") as word_to_idx_file:
            json.dump(word_to_idx, word_to_idx_file)

        # Save label to indexes
        Factory.create_labeled_repository(dataset_conf.labeled_path)
        label_to_idx = Project.create_label_to_idx(
            assistant_conf.get_labelset()
        )
        with open(dataset_conf.labels_to_idx_path, "w") as label_to_idx_file:
            json.dump(label_to_idx, label_to_idx_file)

        # Create Model Config object
        model_conf = ModelConf.from_state(
            project_form_state,
            DEFAULT_NUM_WORDS,
            len(assistant_conf.get_labelset()),
        )

        # Copy implementation of model if it is custom
        if model_conf.is_custom_model_type():
            src_model_implementation = (
                project_form_state.model_implementation_path
            )
            shutil.copy(
                src_model_implementation,
                model_conf.implementation_path,
            )

        # Copy model state if it exists
        if project_form_state.model_state_path:
            shutil.copy(
                project_form_state.model_state_path, model_conf.state_path
            )

        # Create Project Config object
        project_conf = ProjectConf.from_state(
            project_form_state, model_conf, assistant_conf, dataset_conf
        )

        return Project(project_conf, project_form_state.save_path)

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

    def get_labels(self) -> list[LabelData]:
        return self._assistant.labels
