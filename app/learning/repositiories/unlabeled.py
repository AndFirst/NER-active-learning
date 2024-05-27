from typing import List, Set

from app.learning.data_persistence_strategies.base import (
    DataPersistenceStrategy,
)


class UnlabeledSentenceRepository:
    def __init__(self, persistence_strategy: DataPersistenceStrategy) -> None:
        self._persistence_strategy = persistence_strategy
        self._sentences = self._persistence_strategy.load()

    def get_sentence(self, idx: int) -> List[str]:
        return self._sentences[idx]

    def unique_words(self) -> Set[str]:
        return set((word for line in self._sentences for word in line))

    def get_longest_sentence(self) -> List[str]:
        return max(self._sentences, key=len) if self._sentences else []

    def get_sentence_idx(self, sentence: List[str]) -> int:
        return self._sentences.index(sentence)

    def remove_sentence(self, idx: int) -> None:
        self._sentences.pop(idx)
        self._persistence_strategy.save(self._sentences)

    def get_all_sentences(self) -> List[List[str]]:
        return self._sentences

    def save(self) -> None:
        self._persistence_strategy.save(self._sentences)
