from abc import ABC


class LabeledWrapper(ABC):
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
