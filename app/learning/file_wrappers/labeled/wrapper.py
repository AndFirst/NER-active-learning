from abc import ABC, abstractmethod
from typing import List


class LabeledWrapper(ABC):
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._sentences = self.load(file_path)

    def save_sentence(self, sentence: List[str]) -> None:
        self._sentences.append(sentence)

    def get_all_sentences(self) -> List[List[str]]:
        return self._sentences

    def get_longest_sentence(self) -> List[str]:
        return max(self._sentences, key=len) if self._sentences else []

    @abstractmethod
    def load(self, file_path: str) -> List[List[str]]:
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError
