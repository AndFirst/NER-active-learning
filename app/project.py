from __future__ import annotations
from itertools import product
import logging
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
    DEFAULT_UNLABELED_LABEL,
    DEFAULT_UNLABELED_IDX,
)
from app.learning.active_learning import ActiveLearningManager
from app.learning.factory import Factory
from app.learning.models.base import NERModel


class Project:
    """
    The Project class encapsulates the configuration, dataset, model, and assistant for a project.

    :param config: The configuration for the project.
    :type config: ProjectConf
    :param directory: The directory where the project is located.
    :type directory: str
    """

    def __init__(self, config: ProjectConf, directory: str):
        self._config = config
        self._directory = directory
        self._dataset = Factory.create_dataset(config.dataset_conf)
        self._model = Factory.create_model(config.model_conf)
        self._assistant = Factory.create_assistant(self._model, self._dataset, config.assistant_conf)

    @classmethod
    def load(cls, dir_path: str) -> Project:
        """
        Loads a project from a directory.

        :param dir_path: The directory path where the project is located.
        :type dir_path: str
        :return: The loaded project.
        :rtype: Project
        """
        try:
            config = ProjectConf.from_file(f"{dir_path}/project.json")
            return Project(config, dir_path)
        except Exception as e:
            logging.error(f"Failed to load project from {dir_path}: {e}")
            raise

    def save(self) -> None:
        """
        Saves the project to its directory.
        """
        try:
            self._config.save_config(self._directory)
            self._model.save(self._directory + "/model.pth")
            self._dataset.save()
        except Exception as e:
            logging.error(f"Failed to save project: {e}")
            raise

    @classmethod
    def create(cls, project_form_state: ProjectFormState) -> Project:
        """
        Creates a new project based on the provided form state.

        :param project_form_state: The state of the project form.
        :type project_form_state: ProjectFormState
        :return: The created project.
        :rtype: Project
        """
        try:
            cls._create_project_directory(project_form_state.save_path)
            assistant_conf = cls._create_assistant_config(project_form_state)
            dataset_conf = cls._create_dataset_config(project_form_state)
            cls._copy_dataset_to_project_path(project_form_state.dataset_path, dataset_conf.unlabeled_path)
            cls._save_word_to_indexes(dataset_conf.unlabeled_path)
            cls._save_label_to_indexes(
                dataset_conf.labeled_path,
                assistant_conf.get_labelset(),
                dataset_conf.labels_to_idx_path,
            )
            model_conf = cls._create_model_config(project_form_state, len(assistant_conf.get_labelset()))
            cls._copy_model_implementation_if_custom(project_form_state.model_implementation_path, model_conf)
            cls._copy_model_state_if_exists(
                project_form_state.model_state_path, project_form_state.save_path + model_conf.state_path
            )
            project_conf = cls._create_project_config(project_form_state, model_conf, assistant_conf, dataset_conf)
            return Project(project_conf, project_form_state.save_path)
        except Exception as e:
            logging.error(f"Failed to create project: {e}")
            raise

    @staticmethod
    def _create_project_directory(save_path: str) -> None:
        os.makedirs(save_path)

    @staticmethod
    def _create_assistant_config(
        project_form_state: ProjectFormState,
    ) -> AssistantConf:
        return AssistantConf.from_state(project_form_state)

    @staticmethod
    def _create_dataset_config(
        project_form_state: ProjectFormState,
    ) -> DatasetConf:
        return DatasetConf.from_state(project_form_state)

    @staticmethod
    def _copy_dataset_to_project_path(dataset_path: str, unlabeled_path: str) -> None:
        shutil.copy(dataset_path, unlabeled_path)

    @staticmethod
    def _save_word_to_indexes(unlabeled_path: str) -> None:
        Factory.create_unlabeled_repository(unlabeled_path)

    @classmethod
    def _save_label_to_indexes(cls, labeled_path: str, labelset: Set[str], labels_to_idx_path: str) -> None:
        Factory.create_labeled_repository(labeled_path)
        label_to_idx = cls.create_label_to_idx(labelset)
        with open(labels_to_idx_path, "w") as label_to_idx_file:
            json.dump(label_to_idx, label_to_idx_file)

    @staticmethod
    def _create_model_config(project_form_state: ProjectFormState, labelset_length: int) -> ModelConf:
        return ModelConf.from_state(project_form_state, labelset_length)

    @staticmethod
    def _copy_model_implementation_if_custom(model_implementation_path: str, model_conf: ModelConf) -> None:
        if model_conf.is_custom_model_type():
            shutil.copy(model_implementation_path, model_conf.implementation_path)

    @staticmethod
    def _copy_model_state_if_exists(model_state_path: str, state_path: str) -> None:
        if model_state_path:
            shutil.copy(model_state_path, state_path)

    @staticmethod
    def _create_project_config(
        project_form_state: ProjectFormState,
        model_conf: ModelConf,
        assistant_conf: AssistantConf,
        dataset_conf: DatasetConf,
    ) -> ProjectConf:
        return ProjectConf.from_state(project_form_state, model_conf, assistant_conf, dataset_conf)

    @staticmethod
    def create_label_to_idx(labels: Set[str]) -> Dict[str, int]:
        """
        Creates a dictionary mapping labels to indices.

        :param labels: The set of labels.
        :type labels: Set[str]
        :return: The dictionary mapping labels to indices.
        :rtype: Dict[str, int]
        """
        sorted_labels = sorted(list(labels))
        label_to_idx = {DEFAULT_UNLABELED_LABEL: DEFAULT_UNLABELED_IDX}
        label_to_idx.update({f"{prefix}-{label}": idx for idx, (label, prefix) in enumerate(product(sorted_labels, "BI"), 1)})
        return label_to_idx

    @property
    def assistant(self) -> ActiveLearningManager:
        """
        The assistant for the project.

        :return: The assistant.
        :rtype: ActiveLearningManager
        """
        return self._assistant

    @property
    def model(self) -> NERModel:
        """
        The model for the project.

        :return: The model.
        :rtype: NERModel
        """
        return self._model

    @property
    def labels(self) -> list[LabelData]:
        """
        The labels for the project.

        :return: The labels.
        :rtype: list[LabelData]
        """
        return self._assistant.labels
