import os

from app.learning.file_wrappers.labeled.csv import LabeledCsv
from app.learning.file_wrappers.labeled.json import LabeledJson
from app.learning.file_wrappers.unlabeled.csv import UnlabeledCsv
from app.learning.file_wrappers.unlabeled.json import UnlabeledJson
from dataset import Dataset


class DatasetFactory:
    @staticmethod
    def create_dataset(config: dict) -> Dataset:
        unlabeled_path = config["unlabeled_path"]
        labeled_path = config["labeled_path"]

        unlabeled_extension = os.path.splitext(unlabeled_path)[1]
        labeled_extension = os.path.splitext(labeled_path)[1]

        match unlabeled_extension:
            case ".csv":
                unlabeled_file = UnlabeledCsv(unlabeled_path)
            case ".json":
                unlabeled_file = UnlabeledJson(unlabeled_path)
            case _:
                raise ValueError(
                    f"Unsupported extension for unlabeled file: {unlabeled_extension}"
                )
        match labeled_extension:
            case ".csv":
                labeled_file = LabeledCsv(labeled_path)
            case ".json":
                labeled_file = LabeledJson(labeled_path)
            case _:
                raise ValueError(
                    f"Unsupported extension for labeled file: {labeled_extension}"
                )

        return Dataset(
            unlabeled_file=unlabeled_file, labeled_file=labeled_file
        )
