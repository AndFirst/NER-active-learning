import os
from dataclasses import dataclass, field
from itertools import chain
from typing import Tuple, Optional, Any, List, Dict

from app.constants import DEFAULT_UNLABELED_LABEL


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

    def __hash__(self):
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
