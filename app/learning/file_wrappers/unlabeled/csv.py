from typing import List, Set

from .wrapper import UnlabeledWrapper


class UnlabeledCsv(UnlabeledWrapper):
    def __init__(self, file_path: str) -> None:
        if not file_path.endswith(".csv"):
            raise ValueError("File type must be .csv")
        super().__init__(file_path)

    def count_sentences(self) -> int:
        with open(self._file_path, "r") as file:
            sentence_count = 0
            for line in file:
                if line.strip():
                    sentence_count += 1
            return sentence_count

    def unique_words(self) -> Set[str]:
        unique_words = set()
        with open(self._file_path, "r") as file:
            for line in file:
                words = line.strip().split("\t")
                unique_words.update(words)
        return unique_words

    def get_sentence(self, idx: int) -> List[str]:
        sentences = []
        with open(self._file_path, "r") as file:
            for i, line in enumerate(file):
                if i == idx:
                    sentences = line.strip().split("\t")
                    break
        if not sentences:
            raise IndexError("Index out of range")
        return sentences
