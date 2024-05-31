import json
import os
from .base import DataPersistenceStrategy, Sentences


class JsonListStrategy(DataPersistenceStrategy):
    """
    A strategy for persisting data in a JSON file.

    :param file_path: The path to the JSON file.
    :type file_path: str
    :raises ValueError: If the file type is not .json
    """

    def __init__(self, file_path: str) -> None:
        if not file_path.endswith(".json"):
            raise ValueError("File type must be .json")
        self._file_path = file_path

    def load(self) -> Sentences:
        """
        Load sentences from the JSON file.

        :return: A list of sentences, where each sentence is a list of strings.
        :rtype: Sentences
        :raises ValueError: If the JSON format is incorrect.
        """
        if not os.path.isfile(self._file_path):
            with open(self._file_path, "w") as file:
                json.dump([], file)
                return []
        try:
            with open(self._file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list) and all(isinstance(sentence, list) for sentence in data):
                    return data
                else:
                    raise ValueError("JSON format is incorrect. Expected a list of lists.")
        except json.JSONDecodeError:
            return []

    def save(self, sentences: Sentences) -> None:
        """
        Save sentences to the JSON file.

        :param sentences: A list of sentences to save, where each sentence is a list of strings.
        :type sentences: Sentences
        """
        with open(self._file_path, "w", encoding="utf-8") as file:
            json.dump(sentences, file, ensure_ascii=False, indent=4)
