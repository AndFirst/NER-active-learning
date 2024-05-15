from .wrapper import LabeledWrapper


class LabeledJson(LabeledWrapper):
    def __init__(self, file_path: str) -> None:
        assert file_path.endswith(".json")
        super().__init__(file_path)
