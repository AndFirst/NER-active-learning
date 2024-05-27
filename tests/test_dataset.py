from unittest.mock import MagicMock

from app.learning.dataset import Dataset


def test_extract_features_and_labels():
    labeled_file = MagicMock()
    dataset = Dataset(
        labeled_file=labeled_file,
        unlabeled_file=MagicMock(),
        labels_to_idx={"label1": 0, "label2": 1},
        padding_label="",
        padding_idx=0,
        unlabeled_label="",
        unlabeled_idx=0,
        max_sentence_length=10,
    )
    hashed_words, label_indices = dataset._extract_features_and_labels(
        ["word1", "word2", "label1", "label2"]
    )
    assert hashed_words == [
        dataset.hash_string("word1"),
        dataset.hash_string("word2"),
    ]
    assert label_indices == [0, 1]


def test_map_unlabeled_sentence_to_indices():
    unlabeled_file = MagicMock()
    dataset = Dataset(
        labeled_file=MagicMock(),
        unlabeled_file=unlabeled_file,
        labels_to_idx={},
        padding_label="",
        padding_idx=0,
        unlabeled_label="",
        unlabeled_idx=0,
        max_sentence_length=10,
    )
    indices = dataset.map_unlabeled_sentence_to_indices(["word1", "word2"])
    assert indices == [
        dataset.hash_string("word1"),
        dataset.hash_string("word2"),
    ]


def test_map_indices_to_labels():
    labeled_file = MagicMock()
    dataset = Dataset(
        labeled_file=labeled_file,
        unlabeled_file=MagicMock(),
        labels_to_idx={"label1": 0, "label2": 1},
        padding_label="",
        padding_idx=0,
        unlabeled_label="",
        unlabeled_idx=0,
        max_sentence_length=10,
    )
    labels = dataset.map_indices_to_labels([0, 1])
    assert labels == ["label1", "label2"]


def test_move_sentence_to_labeled():
    labeled_file = MagicMock()
    unlabeled_file = MagicMock()
    unlabeled_file.get_sentence_idx.return_value = 0
    dataset = Dataset(
        labeled_file=labeled_file,
        unlabeled_file=unlabeled_file,
        labels_to_idx={"label1": 0, "label2": 1},
        padding_label="",
        padding_idx=0,
        unlabeled_label="",
        unlabeled_idx=0,
        max_sentence_length=10,
    )
    dataset.move_sentence_to_labeled(["word1", "word2", "label1", "label2"])
    labeled_file.save_sentence.assert_called_once_with(
        ["word1", "word2", "label1", "label2"]
    )
    unlabeled_file.remove_sentence.assert_called_once_with(0)


def test_apply_padding():
    labeled_file = MagicMock()
    dataset = Dataset(
        labeled_file=labeled_file,
        unlabeled_file=MagicMock(),
        labels_to_idx={"label1": 0, "label2": 1},
        padding_label="",
        padding_idx=0,
        unlabeled_label="",
        unlabeled_idx=0,
        max_sentence_length=10,
    )
    padded_vector = dataset._apply_padding([1, 2, 3])
    assert padded_vector == [1, 2, 3] + [0] * 7


def test_get_training_data():
    labeled_file = MagicMock()
    labeled_file.get_all_sentences.return_value = [
        ["word1", "word2", "label1", "label2"]
    ]
    dataset = Dataset(
        labeled_file=labeled_file,
        unlabeled_file=MagicMock(),
        labels_to_idx={"label1": 0, "label2": 1},
        padding_label="",
        padding_idx=0,
        unlabeled_label="",
        unlabeled_idx=0,
        max_sentence_length=10,
    )
    features, labels = dataset.get_training_data()
    assert features == [
        [dataset.hash_string("word1"), dataset.hash_string("word2")] + [0] * 8
    ]
    assert labels == [[0, 1] + [0] * 8]


def test_save():
    labeled_file = MagicMock()
    unlabeled_file = MagicMock()
    dataset = Dataset(
        labeled_file=labeled_file,
        unlabeled_file=unlabeled_file,
        labels_to_idx={"label1": 0, "label2": 1},
        padding_label="",
        padding_idx=0,
        unlabeled_label="",
        unlabeled_idx=0,
        max_sentence_length=10,
    )
    dataset.save()
    labeled_file.save.assert_called_once()
    unlabeled_file.save.assert_called_once()


def test_labeled_sentences_count():
    labeled_file = MagicMock()
    labeled_file.get_all_sentences.return_value = [
        ["word1", "word2", "label1", "label2"]
    ]
    dataset = Dataset(
        labeled_file=labeled_file,
        unlabeled_file=MagicMock(),
        labels_to_idx={"label1": 0, "label2": 1},
        padding_label="",
        padding_idx=0,
        unlabeled_label="",
        unlabeled_idx=0,
        max_sentence_length=10,
    )
    assert dataset.labeled_sentences_count == 1


def test_unlabeled_sentences_count():
    unlabeled_file = MagicMock()
    unlabeled_file.get_all_sentences.return_value = [["word1", "word2"]]
    dataset = Dataset(
        labeled_file=MagicMock(),
        unlabeled_file=unlabeled_file,
        labels_to_idx={"label1": 0, "label2": 1},
        padding_label="",
        padding_idx=0,
        unlabeled_label="",
        unlabeled_idx=0,
        max_sentence_length=10,
    )
    assert dataset.unlabeled_sentences_count == 1


def test_hash_string():
    labeled_file = MagicMock()
    dataset = Dataset(
        labeled_file=labeled_file,
        unlabeled_file=MagicMock(),
        labels_to_idx={"label1": 0, "label2": 1},
        padding_label="",
        padding_idx=0,
        unlabeled_label="",
        unlabeled_idx=0,
        max_sentence_length=10,
    )
    hash_value = dataset.hash_string("word1")
    assert isinstance(hash_value, int)
    assert 0 <= hash_value <= 100_001


def test_get_unlabeled_sentence():
    unlabeled_file = MagicMock()
    unlabeled_file.get_sentence.return_value = ["word1", "word2"]
    dataset = Dataset(
        labeled_file=MagicMock(),
        unlabeled_file=unlabeled_file,
        labels_to_idx={},
        padding_label="",
        padding_idx=0,
        unlabeled_label="",
        unlabeled_idx=0,
        max_sentence_length=10,
    )
    result = dataset.get_unlabeled_sentence(0)
    assert result == ["word1", "word2"]


def test_get_weights():
    labeled_file = MagicMock()
    labeled_file.count_labels.return_value = {"label1": 10, "label2": 20}
    dataset = Dataset(
        labeled_file=labeled_file,
        unlabeled_file=MagicMock(),
        labels_to_idx={"label1": 0, "label2": 1},
        padding_label="",
        padding_idx=0,
        unlabeled_label="",
        unlabeled_idx=0,
        max_sentence_length=10,
    )
    weights = dataset.get_weights()
    assert weights == [3.0, 1.5]  # 1.0 / (10/30) and 1.0 / (20/30)
