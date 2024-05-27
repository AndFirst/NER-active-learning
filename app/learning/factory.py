import json
import os

from app.learning.active_learning import ActiveLearningManager
from app.learning.data_persistence_strategies import (
    CsvTabSeparatedStrategy,
    JsonListStrategy,
)
from app.learning.dataset.dataset import Dataset
from app.learning.models.custom import CustomModel
from app.learning.models.lstm import BiLSTMClassifier
from app.learning.models.ner_model import NERModel
from app.data_types import DatasetConf
from app.data_types import ModelConf, AssistantConf
from app.learning.repositiories.labeled import LabeledSentenceRepository
from app.learning.repositiories.unlabeled import UnlabeledSentenceRepository


class Factory:
    @staticmethod
    def create_dataset(cfg: DatasetConf) -> Dataset:
        with open(cfg.labels_to_idx_path, "r") as fh:
            labels_to_idx = json.load(fh)

        with open(cfg.words_to_idx_path, "r") as fh:
            words_to_idx = json.load(fh)

        unlabeled_file = Factory.create_unlabeled_repository(
            cfg.unlabeled_path
        )
        labeled_file = Factory.create_labeled_repository(cfg.labeled_path)

        return Dataset(
            unlabeled_file=unlabeled_file,
            labeled_file=labeled_file,
            labels_to_idx=labels_to_idx,
            words_to_idx=words_to_idx,
            padding_idx=cfg.padding_idx,
            padding_label=cfg.padding_label,
            unlabeled_idx=cfg.unlabeled_idx,
            unlabeled_label=cfg.unlabeled_label,
            max_sentence_length=cfg.max_sentence_length,
        )

    @staticmethod
    def create_model(cfg: ModelConf) -> NERModel:
        common_params = {
            "num_words": cfg.num_words,
            "num_classes": cfg.num_classes,
            "learning_rate": cfg.learning_rate,
        }

        match cfg.type:
            case "BiLSTM":
                model = BiLSTMClassifier(**common_params)
                if os.path.exists(cfg.state_path):
                    model.load_weights(cfg.state_path)
                model.validate_torch_model(cfg.num_words, cfg.num_classes)
                return model

            case "custom":
                model = CustomModel(cfg.implementation_path)
                if os.path.exists(cfg.state_path):
                    model.load_weights(cfg.state_path)
                model.validate_torch_model(cfg.num_words, cfg.num_classes)
                return model

            case _:
                raise ValueError(f"Unsupported model type: {cfg.type}")

    @staticmethod
    def create_assistant(
        model, dataset, config: AssistantConf
    ) -> ActiveLearningManager:
        return ActiveLearningManager(model, dataset, config)

    @staticmethod
    def create_labeled_repository(
        labeled_path: str,
    ) -> LabeledSentenceRepository:
        data_persistence_strategy = Factory.create_data_persistence_strategy(
            labeled_path
        )
        return LabeledSentenceRepository(data_persistence_strategy)

    @staticmethod
    def create_unlabeled_repository(
        unlabeled_path: str,
    ) -> UnlabeledSentenceRepository:
        data_persistence_strategy = Factory.create_data_persistence_strategy(
            unlabeled_path
        )
        return UnlabeledSentenceRepository(data_persistence_strategy)

    @staticmethod
    def create_data_persistence_strategy(file_path: str):
        extension = os.path.splitext(file_path)[1]
        match extension:
            case ".csv":
                return CsvTabSeparatedStrategy(file_path)
            case ".json":
                return JsonListStrategy(file_path)
            case _:
                raise ValueError(f"Unsupported extension: {extension}")
