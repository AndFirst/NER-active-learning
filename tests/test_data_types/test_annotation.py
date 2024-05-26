from app.data_types import Word, LabelData, Annotation
from app.constants import DEFAULT_UNLABELED_LABEL


def test_get_label_with_label():
    words = [Word(word="word1"), Word(word="word2"), Word(word="word3")]
    label = LabelData(label="TestLabel", color=(255, 0, 0, 255))
    annotation = Annotation(words=words, label=label)

    expected_labels = ["B-TestLabel", "I-TestLabel", "I-TestLabel"]
    assert annotation.get_label() == expected_labels


def test_get_label_without_label():
    words = [Word(word="word1"), Word(word="word2")]
    annotation = Annotation(words=words, label=None)

    expected_labels = [DEFAULT_UNLABELED_LABEL]
    assert annotation.get_label() == expected_labels


def test_get_label_single_word_with_label():
    words = [Word(word="word1")]
    label = LabelData(label="SingleLabel", color=(0, 255, 0, 255))
    annotation = Annotation(words=words, label=label)

    expected_labels = ["B-SingleLabel"]
    assert annotation.get_label() == expected_labels


def test_get_label_single_word_without_label():
    words = [Word(word="word1")]
    annotation = Annotation(words=words, label=None)

    expected_labels = [DEFAULT_UNLABELED_LABEL]
    assert annotation.get_label() == expected_labels


def test_get_label_no_words_with_label():
    words = []
    label = LabelData(label="EmptyLabel", color=(0, 0, 255, 255))
    annotation = Annotation(words=words, label=label)

    expected_labels = ["B-EmptyLabel"]
    assert annotation.get_label() == expected_labels


def test_get_label_no_words_without_label():
    words = []
    annotation = Annotation(words=words, label=None)

    expected_labels = [DEFAULT_UNLABELED_LABEL]
    assert annotation.get_label() == expected_labels


def test_equality():
    word1 = Word("word1")
    word2 = Word("word2")
    word3 = Word("word3")
    words1 = [word1, word2]
    label1 = LabelData(label="Label1", color=(255, 0, 0, 255))
    annotation1 = Annotation(words=words1, label=label1)

    words2 = [word1, word2]
    label2 = LabelData(label="Label1", color=(255, 0, 0, 255))
    annotation2 = Annotation(words=words2, label=label2)

    words3 = [word3]
    label3 = LabelData(label="Label2", color=(0, 255, 0, 255))
    annotation3 = Annotation(words=words3, label=label3)

    assert annotation1 == annotation2
    assert annotation1 != annotation3
    assert annotation2 != annotation3
