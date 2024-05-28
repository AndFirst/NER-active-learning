import logging

import pytest

from app.constants import DEFAULT_UNLABELED_LABEL, DEFAULT_UNLABELED_IDX
from unittest.mock import MagicMock, patch, mock_open
from app.data_types import (
    ProjectFormState,
    DatasetConf,
    AssistantConf,
    ModelConf,
    ProjectConf,
    LabelData,
)
from app.learning.active_learning import ActiveLearningManager
from app.learning.factory import Factory
from app.learning.models import NERModel
from app.project import Project


def test_create_label_to_idx():
    labels = {"ORG", "PER", "LOC"}
    expected_label_to_idx = {
        DEFAULT_UNLABELED_LABEL: DEFAULT_UNLABELED_IDX,
        "B-LOC": 1,
        "I-LOC": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-PER": 5,
        "I-PER": 6,
    }
    label_to_idx = Project.create_label_to_idx(labels)
    assert label_to_idx == expected_label_to_idx


@patch("os.makedirs")
def test_create_project_directory(mock_makedirs):
    Project._create_project_directory("test_path")
    mock_makedirs.assert_called_once_with("test_path")


@patch.object(AssistantConf, "from_state")
def test_create_assistant_config(mock_from_state):
    mock_form_state = MagicMock(spec=ProjectFormState)
    Project._create_assistant_config(mock_form_state)
    mock_from_state.assert_called_once_with(mock_form_state)


@patch.object(DatasetConf, "from_state")
def test_create_dataset_config(mock_from_state):
    mock_form_state = MagicMock(spec=ProjectFormState)
    Project._create_dataset_config(mock_form_state)
    mock_from_state.assert_called_once_with(mock_form_state)


@patch("shutil.copy")
def test_copy_dataset_to_project_path(mock_copy):
    Project._copy_dataset_to_project_path(
        "test_dataset_path", "test_unlabeled_path"
    )
    mock_copy.assert_called_once_with(
        "test_dataset_path", "test_unlabeled_path"
    )


@patch("app.project.Factory.create_unlabeled_repository")
def test_save_word_to_indexes(mock_create_unlabeled_repository):
    Project._save_word_to_indexes("test_unlabeled_path")
    mock_create_unlabeled_repository.assert_called_once_with(
        "test_unlabeled_path"
    )


@patch("app.project.Factory.create_labeled_repository")
@patch("builtins.open", new_callable=mock_open)
def test_save_label_to_indexes(mock_file, mock_create_labeled_repository):
    mock_labelset = set()
    Project._save_label_to_indexes(
        "test_labeled_path", mock_labelset, "test_labels_to_idx_path"
    )
    mock_create_labeled_repository.assert_called_once_with("test_labeled_path")


@patch.object(ModelConf, "from_state")
def test_create_model_config(mock_from_state):
    mock_form_state = MagicMock(spec=ProjectFormState)
    Project._create_model_config(mock_form_state, 5)
    mock_from_state.assert_called_once_with(mock_form_state, 5)


@patch("shutil.copy")
def test_copy_model_implementation_if_custom(mock_copy):
    mock_model_conf = MagicMock(spec=ModelConf)
    mock_model_conf.is_custom_model_type.return_value = True
    Project._copy_model_implementation_if_custom(
        "test_model_implementation_path", mock_model_conf
    )
    mock_copy.assert_called_once_with(
        "test_model_implementation_path", mock_model_conf.implementation_path
    )


@patch("shutil.copy")
def test_copy_model_state_if_exists(mock_copy):
    Project._copy_model_state_if_exists(
        "test_model_state_path", "test_state_path"
    )
    mock_copy.assert_called_once_with(
        "test_model_state_path", "test_state_path"
    )


@patch.object(ProjectConf, "from_state")
def test_create_project_config(mock_from_state):
    mock_form_state = MagicMock(spec=ProjectFormState)
    mock_model_conf = MagicMock(spec=ModelConf)
    mock_assistant_conf = MagicMock(spec=AssistantConf)
    mock_dataset_conf = MagicMock(spec=DatasetConf)
    Project._create_project_config(
        mock_form_state,
        mock_model_conf,
        mock_assistant_conf,
        mock_dataset_conf,
    )
    mock_from_state.assert_called_once_with(
        mock_form_state,
        mock_model_conf,
        mock_assistant_conf,
        mock_dataset_conf,
    )


@patch("app.project.Factory.create_dataset", return_value=MagicMock())
@patch("app.project.Factory.create_model", return_value=MagicMock())
@patch("app.project.Factory.create_assistant", return_value=MagicMock())
def test_project_constructor(
    mock_create_dataset, mock_create_model, mock_create_assistant
):
    # Arrange
    mock_config = MagicMock(spec=ProjectConf)
    mock_config.dataset_conf = MagicMock()
    mock_config.assistant_conf = MagicMock()
    mock_config.model_conf = MagicMock()
    mock_directory = "mock_directory"

    # Act
    project = Project(mock_config, mock_directory)

    # Assert
    assert project._config == mock_config
    assert project._directory == mock_directory
    assert project._dataset is not None
    assert project._model is not None
    assert project._assistant is not None


@patch.object(ProjectConf, "from_file")
@patch("app.project.Factory.create_dataset", return_value=MagicMock())
@patch("app.project.Factory.create_model", return_value=MagicMock())
@patch("app.project.Factory.create_assistant", return_value=MagicMock())
def test_load(
    mock_create_assistant,
    mock_create_model,
    mock_create_dataset,
    mock_from_file,
):
    # Arrange
    mock_dir_path = "mock_dir_path"
    mock_config = MagicMock(spec=ProjectConf)
    mock_config.dataset_conf = MagicMock()
    mock_config.assistant_conf = MagicMock()
    mock_config.model_conf = MagicMock()
    mock_from_file.return_value = mock_config

    # Act
    project = Project.load(mock_dir_path)

    # Assert
    mock_from_file.assert_called_once_with(f"{mock_dir_path}/project.json")
    assert project._config == mock_config
    assert project._directory == mock_dir_path


@patch.object(ProjectConf, "from_file")
def test_load_raises_exception(mock_from_file):
    # Arrange
    mock_dir_path = "mock_dir_path"
    mock_from_file.side_effect = Exception("Failed to load project")

    # Act & Assert
    try:
        Project.load(mock_dir_path)
    except Exception as e:
        assert str(e) == "Failed to load project"


@patch.object(ProjectConf, "save_config")
@patch("app.project.Factory.create_dataset", return_value=MagicMock())
@patch("app.project.Factory.create_model", return_value=MagicMock())
@patch("app.project.Factory.create_assistant", return_value=MagicMock())
def test_save(
    mock_create_assistant,
    mock_create_model,
    mock_create_dataset,
    mock_save_config,
):
    # Arrange
    mock_config = MagicMock(spec=ProjectConf)
    mock_config.dataset_conf = MagicMock()
    mock_config.assistant_conf = MagicMock()
    mock_config.model_conf = MagicMock()
    mock_directory = "mock_directory"
    project = Project(mock_config, mock_directory)

    # Act
    project.save()

    # Assert
    mock_config.save_config.assert_called_once_with(mock_directory)
    mock_create_model.return_value.save.assert_called_once_with(
        mock_config.model_conf.state_path
    )
    mock_create_dataset.return_value.save.assert_called_once()


def test_save_project_exception(mocker, caplog):
    # Mock the config, model, and dataset
    mock_config = MagicMock(spec=ProjectConf)
    mock_config.dataset_conf = MagicMock()
    mock_config.assistant_conf = MagicMock()
    mock_config.model_conf = MagicMock()
    mock_model = MagicMock()
    mock_dataset = MagicMock()

    # Mock the Factory methods to prevent actual file operations
    mocker.patch.object(Factory, "create_dataset", return_value=mock_dataset)
    mocker.patch.object(Factory, "create_model", return_value=mock_model)
    mocker.patch.object(Factory, "create_assistant", return_value=MagicMock())

    # Initialize the project with mocked objects
    project = Project(mock_config, "fake_directory")

    # Make the model's save method raise an exception
    mock_model.save.side_effect = Exception("Model save failed")

    with caplog.at_level(logging.ERROR):
        # Call the save method and assert that it raises an exception
        with pytest.raises(Exception, match="Model save failed"):
            project.save()

    # Check that the error was logged
    assert "Failed to save project: Model save failed" in caplog.text


@patch("app.project.Factory")
@patch.object(Project, "_create_project_directory")
@patch.object(Project, "_create_assistant_config")
@patch.object(Project, "_create_dataset_config")
@patch.object(Project, "_copy_dataset_to_project_path")
@patch.object(Project, "_save_word_to_indexes")
@patch.object(Project, "_save_label_to_indexes")
@patch.object(Project, "_create_model_config")
@patch.object(Project, "_copy_model_implementation_if_custom")
@patch.object(Project, "_copy_model_state_if_exists")
@patch.object(Project, "_create_project_config")
def test_create_project(
    mock_create_project_config,
    mock_copy_model_state_if_exists,
    mock_copy_model_implementation_if_custom,
    mock_create_model_config,
    mock_save_label_to_indexes,
    mock_save_word_to_indexes,
    mock_copy_dataset_to_project_path,
    mock_create_dataset_config,
    mock_create_assistant_config,
    mock_create_project_directory,
    mock_factory,
):
    # Arrange
    mock_factory.create_dataset.return_value = MagicMock()
    mock_factory.create_model.return_value = MagicMock()
    mock_factory.create_assistant.return_value = MagicMock()

    mock_form_state = MagicMock(spec=ProjectFormState)
    mock_project_config = MagicMock(spec=ProjectConf)
    mock_project_config.dataset_conf = MagicMock()
    mock_project_config.assistant_conf = MagicMock()
    mock_project_config.model_conf = MagicMock()

    mock_create_project_config.return_value = mock_project_config

    # Act
    project = Project.create(mock_form_state)

    # Assert
    mock_create_project_directory.assert_called_once_with(
        mock_form_state.save_path
    )
    mock_create_assistant_config.assert_called_once_with(mock_form_state)
    mock_create_dataset_config.assert_called_once_with(mock_form_state)
    mock_copy_dataset_to_project_path.assert_called_once()
    mock_save_word_to_indexes.assert_called_once()
    mock_save_label_to_indexes.assert_called_once()
    mock_create_model_config.assert_called_once()
    mock_copy_model_implementation_if_custom.assert_called_once()
    mock_copy_model_state_if_exists.assert_called_once()
    mock_create_project_config.assert_called_once_with(
        mock_form_state,
        mock_create_model_config.return_value,
        mock_create_assistant_config.return_value,
        mock_create_dataset_config.return_value,
    )
    assert isinstance(project, Project)
    assert project._config == mock_project_config
    assert project._directory == mock_form_state.save_path


@patch("app.project.Factory")
@patch.object(Project, "_create_project_directory")
@patch.object(Project, "_create_assistant_config")
@patch.object(Project, "_create_dataset_config")
@patch.object(Project, "_copy_dataset_to_project_path")
@patch.object(Project, "_save_word_to_indexes")
@patch.object(Project, "_save_label_to_indexes")
@patch.object(Project, "_create_model_config")
@patch.object(Project, "_copy_model_implementation_if_custom")
@patch.object(Project, "_copy_model_state_if_exists")
@patch.object(Project, "_create_project_config")
def test_create_project_exception(
    mock_create_project_config,
    mock_copy_model_state_if_exists,
    mock_copy_model_implementation_if_custom,
    mock_create_model_config,
    mock_save_label_to_indexes,
    mock_save_word_to_indexes,
    mock_copy_dataset_to_project_path,
    mock_create_dataset_config,
    mock_create_assistant_config,
    mock_create_project_directory,
    mock_factory,
    mocker,
    caplog,
):
    # Arrange
    mock_factory.create_dataset.return_value = MagicMock()
    mock_factory.create_model.return_value = MagicMock()
    mock_factory.create_assistant.return_value = MagicMock()

    mock_form_state = MagicMock(spec=ProjectFormState)
    mock_project_config = MagicMock(spec=ProjectConf)
    mock_project_config.dataset_conf = MagicMock()
    mock_project_config.assistant_conf = MagicMock()
    mock_project_config.model_conf = MagicMock()

    mock_create_project_config.return_value = mock_project_config

    # Make the _create_project_directory method raise an exception
    mock_create_project_directory.side_effect = Exception(
        "Failed to create project directory"
    )

    with caplog.at_level(logging.ERROR):
        # Call the create method and assert that it raises an exception
        with pytest.raises(
            Exception, match="Failed to create project directory"
        ):
            Project.create(mock_form_state)

    # Check that the error was logged
    assert (
        "Failed to create project: Failed to create project directory"
        in caplog.text
    )


@patch("app.project.Factory")
def test_assistant_property(mock_factory):
    # Arrange
    mock_factory.create_dataset.return_value = MagicMock()
    mock_factory.create_model.return_value = MagicMock()
    mock_factory.create_assistant.return_value = MagicMock()

    mock_config = MagicMock(spec=ProjectConf)
    mock_config.dataset_conf = MagicMock()
    mock_config.model_conf = MagicMock()
    mock_config.assistant_conf = MagicMock()
    mock_directory = "mock_directory"
    project = Project(mock_config, mock_directory)
    expected_assistant = MagicMock(spec=ActiveLearningManager)
    project._assistant = expected_assistant

    # Act
    actual_assistant = project.assistant

    # Assert
    assert actual_assistant == expected_assistant


@patch("app.project.Factory")
def test_model_property(mock_factory):
    # Arrange
    mock_factory.create_dataset.return_value = MagicMock()
    mock_factory.create_model.return_value = MagicMock()
    mock_factory.create_assistant.return_value = MagicMock()
    mock_config = MagicMock(spec=ProjectConf)
    mock_config.dataset_conf = MagicMock()
    mock_config.model_conf = MagicMock()
    mock_config.assistant_conf = MagicMock()
    mock_directory = "mock_directory"
    project = Project(mock_config, mock_directory)
    expected_model = MagicMock(spec=NERModel)
    project._model = expected_model

    # Act
    actual_model = project.model

    # Assert
    assert actual_model == expected_model


@patch("app.project.Factory")
def test_labels_property(mock_factory):
    # Arrange
    mock_factory.create_dataset.return_value = MagicMock()
    mock_factory.create_model.return_value = MagicMock()
    mock_factory.create_assistant.return_value = MagicMock()
    mock_config = MagicMock(spec=ProjectConf)
    mock_config.dataset_conf = MagicMock()
    mock_config.model_conf = MagicMock()
    mock_config.assistant_conf = MagicMock()
    mock_directory = "mock_directory"
    project = Project(mock_config, mock_directory)
    expected_labels = [MagicMock(spec=LabelData)]
    mock_assistant = MagicMock(spec=ActiveLearningManager)
    mock_assistant.labels = expected_labels
    project._assistant = mock_assistant

    # Act
    actual_labels = project.labels

    # Assert
    assert actual_labels == expected_labels
