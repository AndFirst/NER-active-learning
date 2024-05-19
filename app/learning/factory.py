import json
import os

from app.learning.active_learning import ActiveLearningManager
from app.learning.file_wrappers.labeled.csv import LabeledCsv
from app.learning.file_wrappers.labeled.json import LabeledJson
from app.learning.file_wrappers.labeled.wrapper import LabeledWrapper
from app.learning.file_wrappers.unlabeled.csv import UnlabeledCsv
from app.learning.file_wrappers.unlabeled.json import UnlabeledJson
from app.learning.dataset.dataset import Dataset
from app.learning.file_wrappers.unlabeled.wrapper import UnlabeledWrapper
from app.learning.models.custom import CustomModel
from app.learning.models.lstm import BiLSTMClassifier
from app.learning.models.ner_model import NERModel


class Factory:
    @staticmethod
    def create_dataset(config: dict) -> Dataset:
        unlabeled_path = config["unlabeled_path"]
        labeled_path = config["labeled_path"]
        padding_label = config["padding_label"]
        padding_idx = config["padding_idx"]
        unlabeled_label = config["unlabeled_label"]
        unlabeled_idx = config["unlabeled_idx"]

        with open(config["labels_to_idx_path"], "r") as fh:
            labels_to_idx = json.load(fh)

        with open(config["words_to_idx_path"], "r") as fh:
            words_to_idx = json.load(fh)

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
            unlabeled_file=unlabeled_file,
            labeled_file=labeled_file,
            labels_to_idx=labels_to_idx,
            words_to_idx=words_to_idx,
            padding_idx=padding_idx,
            padding_label=padding_label,
            unlabeled_idx=unlabeled_idx,
            unlabeled_label=unlabeled_label,
        )

    @staticmethod
    def create_model(config: dict) -> NERModel:
        model_type = config.get("model_type")
        model_path = config.get("model_path")
        common_params = {
            "num_words": config["num_words"],
            "num_classes": config["num_classes"],
            "learning_rate": config["learning_rate"],
        }
        match model_type:
            case "LSTM":
                model = BiLSTMClassifier(**common_params)
                if os.path.exists(model_path):
                    model.load_weights(model_path)
                return model
            case "custom":
                model = CustomModel(**common_params)
                model.load(model_path)
            case _:
                ...

    @staticmethod
    def create_assistant(
        model, dataset, config: dict
    ) -> ActiveLearningManager:
        return ActiveLearningManager(model, dataset, **config)

    @staticmethod
    def create_labeled_file(labeled_path: str) -> LabeledWrapper:
        labeled_extension = os.path.splitext(labeled_path)[1]
        match labeled_extension:
            case ".csv":
                labeled_file = LabeledCsv(labeled_path)
            case ".json":
                labeled_file = LabeledJson(labeled_path)
            case _:
                raise ValueError(
                    f"Unsupported extension for labeled file: {labeled_extension}"
                )
        return labeled_file

    @staticmethod
    def create_unlabeled_file(unlabeled_path: str) -> UnlabeledWrapper:
        unlabeled_extension = os.path.splitext(unlabeled_path)[1]
        match unlabeled_extension:
            case ".csv":
                unlabeled_file = UnlabeledCsv(unlabeled_path)
            case ".json":
                unlabeled_file = UnlabeledJson(unlabeled_path)
            case _:
                raise ValueError(
                    f"Unsupported extension for unlabeled file: {unlabeled_extension}"
                )
        return unlabeled_file
