from wrapper import LabeledWrapper


class LabeledCsv(LabeledWrapper):
    def __init__(self, file_path: str) -> None:
        assert file_path.endswith(".csv")
        super().__init__(file_path)
