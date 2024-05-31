from unittest.mock import MagicMock
from app.learning.repositiories.unlabeled import UnlabeledSentenceRepository


def test_get_sentence():
    persistence_strategy = MagicMock()
    persistence_strategy.load.return_value = [
        ["word1", "word2"],
        ["word3", "word4"],
    ]
    repo = UnlabeledSentenceRepository(persistence_strategy)
    result = repo.get_sentence(0)
    assert result == ["word1", "word2"]


def test_unique_words():
    persistence_strategy = MagicMock()
    persistence_strategy.load.return_value = [
        ["word1", "word2"],
        ["word3", "word4"],
    ]
    repo = UnlabeledSentenceRepository(persistence_strategy)
    result = repo.unique_words()
    assert result == {"word1", "word2", "word3", "word4"}


def test_get_longest_sentence():
    persistence_strategy = MagicMock()
    persistence_strategy.load.return_value = [
        ["word1", "word2"],
        ["word3", "word4", "word5"],
    ]
    repo = UnlabeledSentenceRepository(persistence_strategy)
    result = repo.get_longest_sentence()
    assert result == ["word3", "word4", "word5"]


def test_get_sentence_idx():
    persistence_strategy = MagicMock()
    persistence_strategy.load.return_value = [
        ["word1", "word2"],
        ["word3", "word4", "word5"],
    ]
    repo = UnlabeledSentenceRepository(persistence_strategy)
    result = repo.get_sentence_idx(["word3", "word4", "word5"])
    assert result == 1


def test_remove_sentence():
    persistence_strategy = MagicMock()
    persistence_strategy.load.return_value = [
        ["word1", "word2"],
        ["word3", "word4", "word5"],
    ]
    repo = UnlabeledSentenceRepository(persistence_strategy)
    repo.remove_sentence(0)
    persistence_strategy.save.assert_called_once_with([["word3", "word4", "word5"]])


def test_get_all_sentences():
    persistence_strategy = MagicMock()
    persistence_strategy.load.return_value = [
        ["word1", "word2"],
        ["word3", "word4", "word5"],
    ]
    repo = UnlabeledSentenceRepository(persistence_strategy)
    result = repo.get_all_sentences()
    assert result == [["word1", "word2"], ["word3", "word4", "word5"]]


def test_save():
    persistence_strategy = MagicMock()
    repo = UnlabeledSentenceRepository(persistence_strategy)
    repo.save()
    persistence_strategy.save.assert_called_once()
