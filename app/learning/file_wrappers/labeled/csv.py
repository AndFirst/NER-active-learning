import csv
from typing import List
import os.path
from .wrapper import LabeledWrapper


class LabeledCsv(LabeledWrapper):
    def __init__(self, file_path: str) -> None:
        assert file_path.endswith(".csv")
        super().__init__(file_path)

    def load(self) -> List[List[str]]:
        if not os.path.isfile(self._file_path):
            with open(self._file_path, "w") as file:
                pass
        with open(self._file_path, "r") as file:
            reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
            sentences = list(reader)
        return sentences

    def save(self) -> None:
        sentences = (
            "\n".join("\t".join(sentence) for sentence in self._sentences)
            + "\n"
        )
        with open(self._file_path, "w") as file:
            file.write(sentences)
