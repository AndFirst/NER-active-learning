import pytest
from unittest.mock import MagicMock

from app.constants import DEFAULT_UNLABELED_LABEL
from app.learning.active_learning import ActiveLearningManager
from app.data_types import Sentence, Annotation, Word, AssistantConf, LabelData


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict_with_confidence.return_value = ([0, 1], [0.9, 0.8])
    model.evaluate_metrics.return_value = {"accuracy": 0.9, "loss": 0.1}
    return model


@pytest.fixture
def mock_dataset():
    dataset = MagicMock()
    dataset.get_unlabeled_sentence.return_value = ["word1", "word2"]
    dataset.map_unlabeled_sentence_to_indices.return_value = [0, 1]
    dataset.map_indices_to_labels.return_value = ["B-label1", "I-label1"]
    dataset.get_training_data.return_value = ([0, 1], [0, 1])
    dataset.get_weights.return_value = [1.0, 1.0]
    dataset.get_labeled_sentences_converted.return_value = ([0, 1], [0, 1])
    dataset.labeled_sentences_count = 5
    dataset.unlabeled_sentences_count = 10
    return dataset


@pytest.fixture
def mock_config():
    config = AssistantConf(
        labels=[LabelData("label1", (0, 0, 0, 0)), LabelData("label2", (0, 0, 0, 0))],
        sampling_batch_size=2,
        epochs=1,
        batch_size=1,
    )
    return config


@pytest.fixture
def active_learning_manager(mock_model, mock_dataset, mock_config):
    return ActiveLearningManager(mock_model, mock_dataset, mock_config)


def test_get_sentence_unannotated(active_learning_manager, mock_dataset):
    mock_dataset.get_unlabeled_sentence.return_value = None
    sentence = active_learning_manager.get_sentence(annotated=False)
    assert sentence is None

    mock_dataset.get_unlabeled_sentence.return_value = ["word1", "word2"]
    sentence = active_learning_manager.get_sentence(annotated=False)
    assert isinstance(sentence, Sentence)
    assert all(isinstance(annotation, Annotation) for annotation in sentence.tokens)
    assert all(annotation.label is None for annotation in sentence.tokens)


def test_get_sentence_annotated(active_learning_manager, mock_dataset):
    mock_dataset.get_unlabeled_sentence.return_value = None
    sentence = active_learning_manager.get_sentence(annotated=True)
    assert sentence is None

    mock_dataset.get_unlabeled_sentence.return_value = ["word1", "word2"]
    sentence = active_learning_manager.get_sentence(annotated=True)
    assert isinstance(sentence, Sentence)
    assert all(isinstance(annotation, Annotation) for annotation in sentence.tokens)
    assert all(annotation.label is not None for annotation in sentence.tokens)


def test_give_feedback(active_learning_manager, mock_dataset):
    sentence = Sentence([Annotation(words=[Word("word1")], label=LabelData("label1", (0, 0, 0, 0)))])
    active_learning_manager.give_feedback(sentence)
    mock_dataset.move_sentence_to_labeled.assert_called_once()
    assert active_learning_manager._annotated_sentences_count == 1


def test_give_feedback_trigger_training(active_learning_manager, mock_dataset, mock_model):
    sentence = Sentence([Annotation(words=[Word("word1")], label=LabelData("label1", (0, 0, 0, 0)))])
    active_learning_manager.give_feedback(sentence)
    active_learning_manager.give_feedback(sentence)
    mock_model.train_async.assert_called_once()


def test_labels(active_learning_manager, mock_config):
    assert active_learning_manager.labels == mock_config.labels


def test_stats(active_learning_manager, mock_dataset, mock_model):
    stats = active_learning_manager.stats
    assert "labeled_sentences_count" in stats
    assert "unlabeled_sentences_count" in stats
    assert "accuracy" in stats
    assert "loss" in stats


def test_create_annotations_default_unlabeled_label(active_learning_manager, mock_config):
    sentence = ["word1", "word2"]
    labels = ["B-label1", DEFAULT_UNLABELED_LABEL]

    annotations = active_learning_manager._create_annotations(sentence, labels)

    assert len(annotations) == 2
    assert annotations[0].label.label == "label1"
    assert annotations[1].label is None
    assert annotations[0].words[0].word == "word1"
    assert annotations[1].words[0].word == "word2"
