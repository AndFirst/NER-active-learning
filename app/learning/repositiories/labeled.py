from collections import Counter
from typing import List, Dict

from app.learning.data_persistence_strategies.base import (
    DataPersistenceStrategy,
)


class LabeledSentenceRepository:
    def __init__(self, persistence_strategy: DataPersistenceStrategy) -> None:
        self._persistence_strategy = persistence_strategy
        self._sentences = self._persistence_strategy.load()

    def save_sentence(self, sentence: List[str]) -> None:
        self._sentences.append(sentence)

    def get_all_sentences(self) -> List[List[str]]:
        return self._sentences

    def get_longest_sentence(self) -> List[str]:
        return max(self._sentences, key=len) if self._sentences else []

    def _get_labels(self) -> List[List[str]]:
        return [sentence[len(sentence) // 2 :] for sentence in self._sentences]

    def count_labels(self) -> Dict[str, int]:
        labels = [label for sublist in self._get_labels() for label in sublist]
        return Counter(labels)

    def save(self) -> None:
        self._persistence_strategy.save(self._sentences)
