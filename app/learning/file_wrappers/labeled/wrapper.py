from abc import ABC, abstractmethod
from typing import List


class LabeledWrapper(ABC):
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path

    @abstractmethod
    def save_sentence(self, sentence: List[str]) -> None:
        pass

    @abstractmethod
    def get_all_sentences(self) -> List[List[str]]:
        raise NotImplementedError

    @abstractmethod
    def get_longest_sentence(self) -> List[str]:
        pass
