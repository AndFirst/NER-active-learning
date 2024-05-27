from collections import Counter
from typing import Dict, List

from app.learning.data_persistence_strategies import DataPersistenceStrategy


class LabeledSentenceRepository:
    """
    A repository for handling labeled sentences.

    :param persistence_strategy: The strategy for data persistence.
    :type persistence_strategy: DataPersistenceStrategy
    """

    def __init__(self, persistence_strategy: DataPersistenceStrategy) -> None:
        """
        Initializes the LabeledSentenceRepository.

        :param persistence_strategy: The strategy for data persistence.
        :type persistence_strategy: DataPersistenceStrategy
        """
        self._persistence_strategy = persistence_strategy
        self._sentences = self._persistence_strategy.load()

    def save_sentence(self, sentence: List[str]) -> None:
        """
        Saves a sentence to the repository.

        :param sentence: The sentence to save.
        :type sentence: List[str]
        """
        self._sentences.append(sentence)

    def get_all_sentences(self) -> List[List[str]]:
        """
        Retrieves all sentences from the repository.

        :return: A list of all sentences.
        :rtype: List[List[str]]
        """
        return self._sentences

    def get_longest_sentence(self) -> List[str]:
        """
        Retrieves the longest sentence from the repository.

        :return: The longest sentence.
        :rtype: List[str]
        """
        return max(self._sentences, key=len) if self._sentences else []

    def _get_labels(self) -> List[List[str]]:
        """
        Retrieves the labels from the sentences in the repository.

        :return: A list of labels.
        :rtype: List[List[str]]
        """
        return [sentence[len(sentence) // 2 :] for sentence in self._sentences]

    def count_labels(self) -> Dict[str, int]:
        """
        Counts the labels in the repository.

        :return: A dictionary where the keys are the labels and the values are the counts of each label.
        :rtype: Dict[str, int]
        """
        labels = [label for sublist in self._get_labels() for label in sublist]
        return Counter(labels)

    def save(self) -> None:
        """
        Saves the current state of the repository.
        """
        self._persistence_strategy.save(self._sentences)
