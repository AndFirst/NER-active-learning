from abc import ABC, abstractmethod
from typing import List, Set


class UnlabeledWrapper(ABC):
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path

    @abstractmethod
    def get_sentence(self, idx: int) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def count_sentences(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def unique_words(self) -> Set[str]:
        raise NotImplementedError
