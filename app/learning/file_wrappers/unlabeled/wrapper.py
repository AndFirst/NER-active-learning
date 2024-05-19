from abc import ABC, abstractmethod
from typing import List, Set


class UnlabeledWrapper(ABC):
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._sentences = self.load(file_path)

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

    @abstractmethod
    def load(self, file_path: str) -> List[List[str]]:
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError
