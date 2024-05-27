from typing import List, Set

from app.learning.data_persistence_strategies.base import (
    DataPersistenceStrategy,
)


class UnlabeledSentenceRepository:
    """
    A repository for handling unlabeled sentences.

    :param persistence_strategy: The strategy for data persistence.
    :type persistence_strategy: DataPersistenceStrategy
    """

    def __init__(self, persistence_strategy: DataPersistenceStrategy) -> None:
        """
        Initializes the UnlabeledSentenceRepository.

        :param persistence_strategy: The strategy for data persistence.
        :type persistence_strategy: DataPersistenceStrategy
        """
        self._persistence_strategy = persistence_strategy
        self._sentences = self._persistence_strategy.load()

    def get_sentence(self, idx: int) -> List[str]:
        """
        Retrieves a sentence from the repository by its index.

        :param idx: The index of the sentence to retrieve.
        :type idx: int
        :return: The sentence as a list of words.
        :rtype: List[str]
        """
        return self._sentences[idx]

    def unique_words(self) -> Set[str]:
        """
        Retrieves all unique words in the repository.

        :return: A set of unique words.
        :rtype: Set[str]
        """
        return set((word for line in self._sentences for word in line))

    def get_longest_sentence(self) -> List[str]:
        """
        Retrieves the longest sentence from the repository.

        :return: The longest sentence.
        :rtype: List[str]
        """
        return max(self._sentences, key=len) if self._sentences else []

    def get_sentence_idx(self, sentence: List[str]) -> int:
        """
        Retrieves the index of a sentence in the repository.

        :param sentence: The sentence to find.
        :type sentence: List[str]
        :return: The index of the sentence.
        :rtype: int
        """
        return self._sentences.index(sentence)

    def remove_sentence(self, idx: int) -> None:
        """
        Removes a sentence from the repository by its index.

        :param idx: The index of the sentence to remove.
        :type idx: int
        """
        self._sentences.pop(idx)
        self._persistence_strategy.save(self._sentences)

    def get_all_sentences(self) -> List[List[str]]:
        """
        Retrieves all sentences from the repository.

        :return: A list of all sentences.
        :rtype: List[List[str]]
        """
        return self._sentences

    def save(self) -> None:
        """
        Saves the current state of the repository.
        """
        self._persistence_strategy.save(self._sentences)
