import csv
import os

from .base import DataPersistenceStrategy, Sentences


class CsvTabSeparatedStrategy(DataPersistenceStrategy):
    """
    A strategy for persisting data in a tab-separated CSV file.

    :param file_path: The path to the CSV file.
    :type file_path: str
    :raises ValueError: If the file type is not .csv
    """

    def __init__(self, file_path: str) -> None:
        if not file_path.endswith(".csv"):
            raise ValueError("File type must be .csv")
        self._file_path = file_path

    def load(self) -> Sentences:
        """
        Load sentences from the CSV file.

        :return: A list of sentences, where each sentence is a list of strings.
        :rtype: Sentences
        """
        if not os.path.isfile(self._file_path):
            with open(self._file_path, "w"):
                return []
        with open(self._file_path, "r") as file:
            reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
            sentences = list(reader)
        return sentences

    def save(self, sentences: Sentences) -> None:
        """
        Save sentences to the CSV file.

        :param sentences: A list of sentences to save, where each sentence is a list of strings.
        :type sentences: Sentences
        """
        with open(self._file_path, "w") as file:
            writer = csv.writer(file, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
            writer.writerows(sentences)
