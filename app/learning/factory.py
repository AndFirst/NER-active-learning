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
from app.data_types import DatasetConf
from app.data_types import ModelConf


class Factory:
    @staticmethod
    def create_dataset(cfg: DatasetConf) -> Dataset:
        with open(cfg.labels_to_idx_path, "r") as fh:
            labels_to_idx = json.load(fh)

        with open(cfg.words_to_idx_path, "r") as fh:
            words_to_idx = json.load(fh)

        unlabeled_file = Factory.create_unlabeled_file(cfg.unlabeled_path)
        labeled_file = Factory.create_labeled_file(cfg.labeled_path)

        return Dataset(
            unlabeled_file=unlabeled_file,
            labeled_file=labeled_file,
            labels_to_idx=labels_to_idx,
            words_to_idx=words_to_idx,
            padding_idx=cfg.padding_idx,
            padding_label=cfg.padding_label,
            unlabeled_idx=cfg.unlabeled_idx,
            unlabeled_label=cfg.unlabeled_label,
        )

    @staticmethod
    def create_model(cfg: ModelConf) -> NERModel:
        common_params = {
            "num_words": cfg.num_words,
            "num_classes": cfg.num_classes,
            "learning_rate": cfg.learning_rate
        }

        match cfg.type:
            case "BiLSTM":
                model = BiLSTMClassifier(**common_params)
                if os.path.exists(cfg.state_path):
                    model.load_weights(cfg.state_path)
                model.validate_torch_model(
                    cfg.num_words, cfg.num_classes
                )
                return model

            case "custom":
                model = CustomModel(cfg.implementation_path)
                if os.path.exists(cfg.state_path):
                    model.load_weights(cfg.state_path)
                model.validate_torch_model(
                    cfg.num_words, cfg.num_classes
                )
                return model

            case _:
                ...

    @staticmethod
    def create_assistant(model, dataset,
                         config: dict) -> ActiveLearningManager:
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
