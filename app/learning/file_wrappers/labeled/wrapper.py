from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Dict


class LabeledWrapper(ABC):
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._sentences = self.load()

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

    @abstractmethod
    def load(self) -> List[List[str]]:
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError
