import csv
from typing import List, Set

from .wrapper import UnlabeledWrapper


class UnlabeledCsv(UnlabeledWrapper):
    def __init__(self, file_path: str) -> None:
        if not file_path.endswith(".csv"):
            raise ValueError("File type must be .csv")
        super().__init__(file_path)

    def get_longest_sentence(self) -> List[str]:
        with open(self._file_path, "r") as file:
            reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
            sentences = list(reader)
        return max(sentences, key=len) if sentences else []

    def get_sentence_idx(self, sentence: List[str]) -> int:
        with open(self._file_path, "r") as file:
            reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
            sentences = list(reader)
        return sentences.index(sentence)

    def remove_sentence(self, idx: int) -> None:
        with open(self._file_path, "r") as file:
            reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
            sentences = list(reader)
        sentences.pop(idx)
        with open(self._file_path, "w") as file:
            file.truncate(0)
            for sentence in sentences:
                file.write("\t".join(sentence) + "\n")

    def count_sentences(self) -> int:
        with open(self._file_path, "r") as file:
            reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
            sentences = list(reader)
        return len(sentences)

    def unique_words(self) -> Set[str]:
        with open(self._file_path, "r") as file:
            reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
            unique_words = set((word for line in reader for word in line))
        return unique_words

    def get_sentence(self, idx: int) -> List[str]:
        with open(self._file_path, "r") as file:
            reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
            sentences = list(reader)
        return sentences[idx]
