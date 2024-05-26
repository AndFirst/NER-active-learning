import csv
from typing import List

from .wrapper import UnlabeledWrapper


class UnlabeledCsv(UnlabeledWrapper):
    def __init__(self, file_path: str) -> None:
        if not file_path.endswith(".csv"):
            raise ValueError("File type must be .csv")
        super().__init__(file_path)

    def load(self) -> List[List[str]]:
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