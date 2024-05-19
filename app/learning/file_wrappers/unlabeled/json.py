import json
from typing import List

from .wrapper import UnlabeledWrapper


class UnlabeledJson(UnlabeledWrapper):

    def __init__(self, file_path: str) -> None:
        if not file_path.endswith(".json"):
            raise ValueError("File type must be .json")
        super().__init__(file_path)

    def load(self, file_path: str) -> List[List[str]]:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, list) and all(
                isinstance(sentence, list) for sentence in data
            ):
                self._sentences = data
                return data
            else:
                raise ValueError(
                    "JSON format is incorrect. Expected a list of lists."
                )

    def save(self) -> None:
        with open(self._file_path, "w", encoding="utf-8") as file:
            json.dump(self._sentences, file, ensure_ascii=False, indent=4)
