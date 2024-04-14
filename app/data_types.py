from dataclasses import dataclass, field
from typing import Tuple, Optional, Any, List, Dict


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


@dataclass
class Word:
    word: str

    def __hash__(self):
        return hash(self.word)


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


@dataclass
class ProjectData:
    name: str = ""
    description: str = ""
    dataset_path: str = ""
    labels: List[LabelData] = field(default_factory=list)

    def to_dict(self):
        data = self.__dict__
        data["labels"] = [label.to_dict() for label in data["labels"]]
        return data
