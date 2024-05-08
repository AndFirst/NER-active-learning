import json
from typing import Set, List

from .wrapper import UnlabeledWrapper


class UnlabeledJson(UnlabeledWrapper):
    def __init__(self, file_path: str) -> None:
        if not file_path.endswith(".json"):
            raise ValueError("File type must be .json")
        super().__init__(file_path)

    def get_sentence(self, idx: int) -> List[str]:
        with open(self._file_path, "r") as file:
            data = json.load(file)
            if idx < 0 or idx >= len(data):
                raise IndexError("Index out of range")
            return data[idx]

    def count_sentences(self) -> int:
        with open(self._file_path, "r") as file:
            data = json.load(file)
            return len(data)

    def unique_words(self) -> Set[str]:
        unique_words = set()
        with open(self._file_path, "r") as file:
            data = json.load(file)
            for sentence in data:
                unique_words.update(sentence)
        return unique_words
