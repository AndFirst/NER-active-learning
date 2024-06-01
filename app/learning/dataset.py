import hashlib
from typing import Dict, List, Tuple

from app.learning.repositiories import (
    LabeledSentenceRepository,
    UnlabeledSentenceRepository,
)


class Dataset:
    """
    A class for handling a dataset for a machine learning model.

    :param labeled_file: A repository for labeled sentences.
    :type labeled_file: LabeledSentenceRepository
    :param unlabeled_file: A repository for unlabeled sentences.
    :type unlabeled_file: UnlabeledSentenceRepository
    :param labels_to_idx: A dictionary mapping labels to indices.
    :type labels_to_idx: Dict[str, int]
    :param padding_label: The label used for padding.
    :type padding_label: str
    :param padding_idx: The index used for padding.
    :type padding_idx: int
    :param unlabeled_label: The label used for unlabeled data.
    :type unlabeled_label: str
    :param unlabeled_idx: The index used for unlabeled data.
    :type unlabeled_idx: int
    :param max_sentence_length: The maximum length of a sentence.
    :type max_sentence_length: int
    """

    def __init__(
        self,
        labeled_file: LabeledSentenceRepository,
        unlabeled_file: UnlabeledSentenceRepository,
        labels_to_idx: Dict[str, int],
        padding_label: str,
        padding_idx: int,
        unlabeled_label: str,
        unlabeled_idx: int,
        max_sentence_length: int,
    ):
        self._labeled_file = labeled_file
        self._unlabeled_file = unlabeled_file

        self._labels_to_idx: Dict[str, int] = labels_to_idx
        self._idx_to_labels: Dict[int, str] = {v: k for k, v in labels_to_idx.items()}

        self._padding_label: str = padding_label
        self._padding_idx: int = padding_idx

        self._unlabeled_label: str = unlabeled_label
        self._unlabeled_idx: int = unlabeled_idx

        self._max_sentence_length: int = max_sentence_length

    def count_labels(self, minimum_one: bool = True) -> Dict[str, int]:
        """
        Counts the labels in the dataset.

        This method counts the labels in the dataset by initializing a dictionary with keys from
        `_labels_to_idx` and values set to 1 if minimum_one is True, else 0.
        Then it updates this dictionary with the counts from the labeled file.

        :param minimum_one: Whether to set the initial count to at least 1.
        :type minimum_one: bool
        :return: A dictionary where the keys are the labels and the values are the counts of each label.
        :rtype: Dict[str, int]
        """
        initial_count = 1 if minimum_one else 0
        label_counts = {label: initial_count for label in self._labels_to_idx.keys()}
        label_counts.update(self._labeled_file.count_labels())
        return label_counts

    def get_weights(self) -> List[float]:
        """
        Calculates the weights for each label in the dataset.

        This method calculates the weights by first counting the labels in the dataset and then dividing 1.0 by the
        proportion of each label in the dataset. The labels are sorted according to their indices in `_labels_to_idx`.

        :return: A list of weights corresponding to each label in the dataset. The weights are in the same order as the
            labels in `_labels_to_idx`.
        :rtype: List[float]
        """
        label_counts = self.count_labels()
        total_count = sum(label_counts.values())

        return [1.0 / (label_counts[label] / total_count) for label in sorted(self._labels_to_idx, key=self._labels_to_idx.get)]

    def get_unlabeled_sentence(self, idx: int) -> List[str]:
        """
        This method retrieves an unlabeled sentence from the dataset by its index.

        :param idx: The index of the sentence to retrieve.
        :type idx: int
        :return: The unlabeled sentence as a list of words.
        :rtype: List[str]
        """
        return self._unlabeled_file.get_sentence(idx)

    def _extract_features_and_labels(self, sentence: List[str]) -> Tuple[List[int], List[int]]:
        """
        Extracts features and labels from a sentence.

        This method takes a sentence, which is a list of words followed by their labels, and returns two lists:
        one with the hashed words and one with the indices of the labels.

        :param sentence: The sentence to extract features and labels from. Must have an even number of words.
        :type sentence: List[str]
        :return: A tuple containing a list of hashed words and a list of label indices.
        :rtype: Tuple[List[int], List[int]]
        """
        assert len(sentence) % 2 == 0, "Sentence must have an even number of words."

        middle = len(sentence) // 2
        words, labels = sentence[:middle], sentence[middle:]

        hashed_words = [self.hash_string(word) for word in words]
        label_indices = [self._labels_to_idx[label] for label in labels]

        return hashed_words, label_indices

    def map_unlabeled_sentence_to_indices(self, sentence: List[str]) -> List[int]:
        """
        Maps an unlabeled sentence to a list of indices.

        This method takes an unlabeled sentence, which is a list of words, and returns a list of indices.
        Each word in the sentence is hashed to a unique index using the `hash_string` method.

        :param sentence: The sentence to map to indices.
        :type sentence: List[str]
        :return: A list of indices corresponding to the words in the sentence.
        :rtype: List[int]
        """
        return [self.hash_string(word) for word in sentence]

    def map_indices_to_labels(self, indices: List[int]) -> List[str]:
        """
        Maps a list of indices to their corresponding labels.

        This method takes a list of indices and returns a list of labels. Each index in the input list is mapped to a
        label using the `_idx_to_labels` dictionary.

        :param indices: The list of indices to map to labels.
        :type indices: List[int]
        :return: A list of labels corresponding to the input indices.
        :rtype: List[str]
        """
        return [self._idx_to_labels[idx] for idx in indices]

    def move_sentence_to_labeled(self, sentence: List[str]) -> None:
        """
        Moves a sentence from the unlabeled file to the labeled file.

        This method takes a sentence, which is a list of words followed by their labels, and moves it from the unlabeled file
        to the labeled file. The sentence is removed from the unlabeled file and saved in the labeled file.

        :param sentence: The sentence to move. Must have an even number of words.
        :type sentence: List[str]
        """
        assert len(sentence) % 2 == 0, "Sentence must have an even number of words."
        middle = len(sentence) // 2
        words = sentence[:middle]
        idx = self._unlabeled_file.get_sentence_idx(words)
        self._labeled_file.save_sentence(sentence)
        self._unlabeled_file.remove_sentence(idx)

    def _apply_padding(self, vector: List[int], max_length: int = None) -> List[int]:
        """
        Applies padding to a vector.

        This method takes a vector and applies padding to it. If the length of the vector is less than `_max_sentence_length`,
        it appends `_padding_idx` to the vector until its length equals `_max_sentence_length`. If the length of the vector
        is greater than `_max_sentence_length`, it truncates the vector to `_max_sentence_length`.

        :param vector: The vector to apply padding to.
        :type vector: List[int]
        :return: The padded or truncated vector.
        :rtype: List[int]
        """
        return vector[:max_length] + [self._padding_idx] * (self._max_sentence_length - len(vector))

    def get_training_data(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Retrieves the training data from the dataset.

        This method retrieves all sentences from the labeled file, extracts features and labels from each sentence,
        applies padding to the features and labels, and returns them as two lists.

        The features are the hashed words from the sentences and the labels are the indices of the labels.

        :return: A tuple containing a list of features and a list of labels. Each feature and label is a list of integers.
        :rtype: Tuple[List[List[int]], List[List[int]]]
        """
        sentences = self._labeled_file.get_all_sentences()

        features, labels = zip(*[self._extract_features_and_labels(sentence) for sentence in sentences])
        features = [self._apply_padding(feature) for feature in features]
        labels = [self._apply_padding(label) for label in labels]
        return features, labels

    def save(self) -> None:
        """
        Saves the current state of the dataset.

        This method saves the current state of the dataset by calling the `save` method on both the labeled and unlabeled
        file repositories. This ensures that any changes made to the dataset are persisted.

        :return: None
        """
        self._labeled_file.save()
        self._unlabeled_file.save()

    @property
    def labeled_sentences_count(self) -> int:
        """
        Returns the number of labeled sentences in the dataset.

        This method retrieves all sentences from the labeled file and returns their count.

        :return: The number of labeled sentences in the dataset.
        :rtype: int
        """
        return len(self._labeled_file.get_all_sentences())

    @property
    def unlabeled_sentences_count(self) -> int:
        """
        Returns the number of unlabeled sentences in the dataset.

        This method retrieves all sentences from the unlabeled file and returns their count.

        :return: The number of unlabeled sentences in the dataset.
        :rtype: int
        """
        return len(self._unlabeled_file.get_all_sentences())

    @staticmethod
    def hash_string(s: str, num_hashes: int = 100_000) -> int:
        """
        Hashes a string to a unique index.

        This method takes a string and a number of hashes, hashes the string using SHA-256, and returns the hash modulo
        the number of hashes plus one. This ensures that the hash is a unique index between 0 and `num_hashes`.

        :param s: The string to hash.
        :type s: str
        :param num_hashes: The number of hashes.
        :type num_hashes: int
        :return: The hash of the string as a unique index.
        :rtype: int
        """
        return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (num_hashes + 1)

    def get_labeled_sentences_converted(
        self,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Retrieves the labeled sentences from the dataset.

        This method retrieves all sentences from the labeled file, extracts features and labels from each sentence,
        applies padding to the features and labels, and returns them as two lists.

        The features are the hashed words from the sentences and the labels are the indices of the labels.

        :return: A tuple containing a list of features and a list of labels.
            Each feature and label is a list of integers.
        :rtype: Tuple[List[List[int]], List[List[int]]]
        """
        sentences = self._labeled_file.get_all_sentences()

        features, labels = zip(*[self._extract_features_and_labels(sentence) for sentence in sentences])

        return features, labels
