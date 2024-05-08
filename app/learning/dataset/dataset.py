from abc import ABC, abstractmethod
from typing import List

from app.learning.file_wrappers.labeled.wrapper import LabeledWrapper
from app.learning.file_wrappers.unlabeled.wrapper import UnlabeledWrapper


class Dataset(ABC):
    def __init__(
        self, labeled_file: LabeledWrapper, unlabeled_file: UnlabeledWrapper
    ):
        self._labeled_file = labeled_file
        self._unlabeled_file = unlabeled_file

    @abstractmethod
    def count_unlabeled_sentences(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def count_labeled_sentences(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def move_sentence_to_labeled(self, sentence: List[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def unique_words_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_longest_sentence_length(self) -> int:
        raise NotImplementedError
