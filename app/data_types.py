from dataclasses import dataclass, field
from typing import Tuple, Optional, Any, List, Dict, IO


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


@dataclass
class Sentence:
    tokens: list[Annotation]

    def get_left_neighbor(self, word: Word) -> Optional[Word]:
        if word == self.tokens[0].words[0]:
            return None
        word_parent = self.get_word_parent(word)
        word_parent_index = self.tokens.index(word_parent)
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
        word_parent_index = self.tokens.index(word_parent)
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

    def to_csv(self, fh: IO[str]) -> None:
        # @TODO Add B-label, I-label to recognize multilabel sentence
        labels = [
            token.label.label if token.label else "O"
            for token in self.tokens
            for word in token.words
        ]
        words = [word.word for token in self.tokens for word in token.words]

        merged = words + ["<END_SENTENCE>"] + labels
        row = ",".join(merged)
        fh.write(row + "\n")


@dataclass
class ProjectData:
    name: str = ""
    description: str = ""
    save_path: str = ""
    dataset_path: str = ""
    labels: List[LabelData] = field(default_factory=list)

    def to_dict(self):
        data = self.__dict__
        data["labels"] = [label.to_dict() for label in data["labels"]]
        return data

    @classmethod
    def from_dict(cls, data_dict):
        label_dicts = data_dict.pop("labels", [])
        labels = [LabelData(**label_dict) for label_dict in label_dicts]
        return cls(labels=labels, **data_dict)
