import csv
from typing import List

from .wrapper import LabeledWrapper


class LabeledCsv(LabeledWrapper):
    def __init__(self, file_path: str) -> None:
        assert file_path.endswith(".csv")
        super().__init__(file_path)

    def save_sentence(self, sentence: List[str]) -> None:
        with open(self._file_path, "a") as file:
            file.write("\t".join(sentence) + "\n")

    def get_all_sentences(self) -> List[List[str]]:
        with open(self._file_path, "r") as file:
            reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
            sentences = list(reader)
        return sentences

    def get_longest_sentence(self) -> List[str]:
        with open(self._file_path, "r") as file:
            reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
            sentences = list(reader)
        return max(sentences, key=len) if sentences else []
