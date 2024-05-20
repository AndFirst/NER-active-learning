import json
from typing import List

from .wrapper import LabeledWrapper


class LabeledJson(LabeledWrapper):
    def __init__(self, file_path: str) -> None:
        assert file_path.endswith(".json")
        super().__init__(file_path)

    def load(self) -> List[List[str]]:
        try:
            with open(self._file_path, "r", encoding="utf-8") as file:
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
        except json.JSONDecodeError:
            return []

    def save(self) -> None:
        with open(self._file_path, "w", encoding="utf-8") as file:
            json.dump(self._sentences, file, ensure_ascii=False, indent=4)
