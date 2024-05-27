from unittest.mock import MagicMock
from app.learning.repositiories.labeled import LabeledSentenceRepository


def test_save_sentence():
    persistence_strategy = MagicMock()
    persistence_strategy.load.return_value = []
    repo = LabeledSentenceRepository(persistence_strategy)
    repo.save_sentence(["word1", "label1"])
    assert repo.get_all_sentences() == [["word1", "label1"]]


def test_get_all_sentences():
    persistence_strategy = MagicMock()
    persistence_strategy.load.return_value = [
        ["word1", "label1"],
        ["word2", "label2"],
    ]
    repo = LabeledSentenceRepository(persistence_strategy)
    assert repo.get_all_sentences() == [
        ["word1", "label1"],
        ["word2", "label2"],
    ]


def test_get_longest_sentence():
    persistence_strategy = MagicMock()
    persistence_strategy.load.return_value = [
        ["word1", "label1"],
        ["word2", "label2", "word3", "label3"],
    ]
    repo = LabeledSentenceRepository(persistence_strategy)
    assert repo.get_longest_sentence() == [
        "word2",
        "label2",
        "word3",
        "label3",
    ]


def test_count_labels():
    persistence_strategy = MagicMock()
    persistence_strategy.load.return_value = [
        ["word1", "label1"],
        ["word2", "label2"],
        ["word3", "label1"],
    ]
    repo = LabeledSentenceRepository(persistence_strategy)
    assert repo.count_labels() == {"label1": 2, "label2": 1}


def test_save():
    persistence_strategy = MagicMock()
    repo = LabeledSentenceRepository(persistence_strategy)
    repo.save()
    persistence_strategy.save.assert_called_once()
