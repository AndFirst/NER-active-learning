from __future__ import annotations
import os
from dataclasses import dataclass, field
from itertools import chain
from typing import Tuple, Optional, Any, List, Dict
import json
from app.exceptions import NoLabelFoundError
from app.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_DROPOUT,
    DEFAULT_INPUT_EXTENSION,
    DEFAULT_OUTPUT_EXTENSION,
    DEFAULT_PADDING_LABEL,
    DEFAULT_PADDING_IDX,
    DEFAULT_UNLABELED_LABEL,
    DEFAULT_UNLABELED_IDX,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SAMPLING_BATCH_SIZE,
    DEFAULT_NUM_WORDS,
    DEFAULT_MAX_SENTENCE_LENGTH,
)


@dataclass
class LabelData:
    label: str
    color: Tuple[int, int, int, int]

    def is_empty(self) -> bool:
        return not self.label.strip()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, LabelData):
            return False
        return self.label.lower() == other.label.lower()

    def __hash__(self) -> int:
        return hash(self.label.lower())

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "color": self.color}


@dataclass(order=True, frozen=True)
class Word:
    word: str

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: Any) -> bool:
        return id(self) == id(other)


@dataclass
class Annotation:
    words: List[Word]
    label: Optional[LabelData]

    def get_label(self) -> List[str]:
        if self.label is None:
            return [DEFAULT_UNLABELED_LABEL]
        else:
            label_text = self.label.label
            labels = ["B-" + label_text] + ["I-" + label_text] * (
                len(self.words) - 1
            )
            return labels


@dataclass
class Sentence:
    tokens: list[Annotation]

    def get_left_neighbor(self, word: Word) -> Optional[Word]:
        if word == self.tokens[0].words[0]:
            return None
        word_parent = self.get_word_parent(word)
        try:
            word_parent_index = self.tokens.index(word_parent)
        except ValueError:
            return None
        if word == word_parent.words[0]:
            left_token = self.tokens[word_parent_index - 1]
            return left_token.words[-1]
        else:
            word_index = word_parent.words.index(word)
            return word_parent.words[word_index - 1]

    def get_right_neighbor(self, word: Word) -> Optional[Word]:
        if word == self.tokens[-1].words[-1]:
            return None
        word_parent = self.get_word_parent(word)
        try:
            word_parent_index = self.tokens.index(word_parent)
        except ValueError:
            return None
        if word == word_parent.words[-1]:
            right_token = self.tokens[word_parent_index + 1]
            return right_token.words[0]
        else:
            word_index = word_parent.words.index(word)
            return word_parent.words[word_index + 1]

    def get_word_parent(self, word: Word) -> Optional[Annotation]:
        for token in self.tokens:
            if word in token.words:
                return token

    def to_list(self) -> List[str]:
        labels = list(
            chain.from_iterable(token.get_label() for token in self.tokens)
        )
        words = [word.word for token in self.tokens for word in token.words]
        return words + labels


@dataclass
class ProjectFormState:
    name: str = ""
    description: str = ""
    save_path: str = ""
    output_extension: str = None
    dataset_path: str = ""
    labels: List[LabelData] = field(default_factory=list)
    model_type: str = ""
    model_state_path: str = None
    model_implementation_path: str = None

    @property
    def input_extension(self) -> str:
        return os.path.splitext(self.dataset_path)[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "description": self.description,
            "save_path": self.save_path,
            "dataset_path": self.dataset_path,
            "labels": [label.to_dict() for label in self.labels],
            "model_state_path": self.model_state_path,
            "model_implementation_path": self.model_implementation_path,
            "input_extension": self.input_extension,
            "output_extension": self.output_extension,
        }

    def get(self, prop: str, default: Any) -> Any:
        if prop in self.to_dict():
            val = self.to_dict()[prop]
            if val != "" and val is not None:
                return val
        return default


@dataclass
class DatasetConf:
    unlabeled_path: str
    labeled_path: str
    labels_to_idx_path: str
    padding_label: str
    padding_idx: int
    unlabeled_label: str
    unlabeled_idx: int
    max_sentence_length: int

    @classmethod
    def from_state(cls, project_form_state: ProjectFormState) -> DatasetConf:
        input_extension = project_form_state.get(
            "input_extension", DEFAULT_INPUT_EXTENSION
        )
        output_extension = project_form_state.get(
            "output_extension", DEFAULT_OUTPUT_EXTENSION
        )
        return DatasetConf(
            f"{project_form_state.save_path}/unlabeled{input_extension}",
            f"{project_form_state.save_path}/labeled{output_extension}",
            f"{project_form_state.save_path}/labels_to_idx.json",
            DEFAULT_PADDING_LABEL,
            DEFAULT_PADDING_IDX,
            DEFAULT_UNLABELED_LABEL,
            DEFAULT_UNLABELED_IDX,
            DEFAULT_MAX_SENTENCE_LENGTH,
        )

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> DatasetConf:
        return DatasetConf(
            dictionary["unlabeled_path"],
            dictionary["labeled_path"],
            dictionary["labels_to_idx_path"],
            dictionary["padding_label"],
            dictionary["padding_idx"],
            dictionary["unlabeled_label"],
            dictionary["unlabeled_idx"],
            dictionary["max_sentence_length"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unlabeled_path": self.unlabeled_path,
            "labeled_path": self.labeled_path,
            "labels_to_idx_path": self.labels_to_idx_path,
            "padding_label": self.padding_label,
            "padding_idx": self.padding_idx,
            "unlabeled_label": self.unlabeled_label,
            "unlabeled_idx": self.unlabeled_idx,
            "max_sentence_length": self.max_sentence_length,
        }

    def get(self, prop: str, default: Any) -> Any:
        if prop in self.to_dict():
            return self.to_dict()[prop]
        else:
            return default


@dataclass
class AssistantConf:
    batch_size: int
    epochs: int
    sampling_batch_size: int
    labels: List[LabelData] = field(default_factory=list)

    @classmethod
    def from_state(cls, project_form_state: ProjectFormState) -> AssistantConf:
        return AssistantConf(
            project_form_state.get("batch_size", DEFAULT_BATCH_SIZE),
            project_form_state.get("epochs", DEFAULT_EPOCHS),
            project_form_state.get(
                "sampling_batch_size", DEFAULT_SAMPLING_BATCH_SIZE
            ),
            project_form_state.labels,
        )

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> AssistantConf:
        return AssistantConf(
            dictionary["batch_size"],
            dictionary["epochs"],
            dictionary["sampling_batch_size"],
            [
                LabelData(label["label"], label["color"])
                for label in dictionary["labels"]
            ],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "sampling_batch_size": self.sampling_batch_size,
            "labels": [label.to_dict() for label in self.labels],
        }

    def get_label(self, label_name: str) -> LabelData:
        for label in self.labels:
            if label.label == label_name:
                return label
        raise NoLabelFoundError

    def get_labelset(self) -> set[str]:
        return {label.to_dict()["label"] for label in self.labels}

    def get(self, prop: str, default: Any) -> Any:
        if prop in self.to_dict():
            return self.to_dict()[prop]
        else:
            return default


@dataclass
class ModelConf:
    type: str
    state_path: str
    dropout: float
    learning_rate: float
    num_words: int
    num_labels: int
    num_classes: int
    implementation_path: str = ""

    @classmethod
    def from_state(
        cls, project_form_state: ProjectFormState, n_labels: int
    ) -> ModelConf:
        impl_path = ""
        if project_form_state.model_type == "custom":
            model_path = "app/learning/models/custom_model_"
            impl_path = f"{model_path}{project_form_state.name}.py"
        return ModelConf(
            project_form_state.model_type,
            f"{project_form_state.save_path}/model.pth",
            project_form_state.get("dropout", DEFAULT_DROPOUT),
            project_form_state.get("learning_rate", DEFAULT_LEARNING_RATE),
            project_form_state.get("num_words", DEFAULT_NUM_WORDS),
            n_labels,
            n_labels * 2 + 1,
            impl_path,
        )

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> ModelConf:
        return ModelConf(
            dictionary["type"],
            dictionary["state_path"],
            dictionary["dropout"],
            dictionary["learning_rate"],
            dictionary["num_words"],
            dictionary["num_labels"],
            dictionary["num_classes"],
            dictionary["implementation_path"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "state_path": self.state_path,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "num_words": self.num_words,
            "num_labels": self.num_labels,
            "num_classes": self.num_classes,
            "implementation_path": self.implementation_path,
        }

    def is_custom_model_type(self) -> bool:
        return self.type == "custom"

    def get(self, prop: str, default: Any) -> Any:
        if prop in self.to_dict():
            return self.to_dict()[prop]
        else:
            return default


@dataclass
class ProjectConf:
    name: str
    description: str
    model_conf: ModelConf
    assistant_conf: AssistantConf
    dataset_conf: DatasetConf

    @classmethod
    def from_state(
        cls,
        project_form_state: ProjectFormState,
        m_conf: ModelConf,
        a_conf: AssistantConf,
        d_conf: DatasetConf,
    ) -> ProjectConf:
        return ProjectConf(
            project_form_state.name,
            project_form_state.description,
            m_conf,
            a_conf,
            d_conf,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model_conf.to_dict(),
            "assistant": self.assistant_conf.to_dict(),
            "dataset": self.dataset_conf.to_dict(),
        }

    def save_config(self, path: str) -> None:
        with open(f"{path}/project.json", "w") as project_cfg_file:
            json.dump(self.to_dict(), project_cfg_file)

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> ProjectConf:
        m_conf = ModelConf.from_dict(dictionary["model"])
        a_conf = AssistantConf.from_dict(dictionary["assistant"])
        d_conf = DatasetConf.from_dict(dictionary["dataset"])
        return ProjectConf(
            dictionary["name"],
            dictionary["description"],
            m_conf,
            a_conf,
            d_conf,
        )

    @classmethod
    def from_file(cls, path: str) -> ProjectConf:
        if not os.path.isfile(path):
            raise FileNotFoundError("Project configuration file not found.")
        with open(path, "r") as cfg_file:
            cfg = json.load(cfg_file)
        return ProjectConf.from_dict(cfg)

    def get(self, prop: str, default: Any) -> Any:
        if prop in self.to_dict():
            return self.to_dict()[prop]
        else:
            return default
