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
        return {
            "label": self.label,
            "color": self.color
        }


@dataclass
class ProjectData:
    name: str = ""
    description: str = ""
    dataset_path: str = ""
    labels: List[LabelData] = field(default_factory=list)

    def to_dict(self):
        data = self.__dict__
        data['labels'] = [label.to_dict() for label in data['labels']]
        return data


@dataclass
class Annotation:
    words: List[str]
    label: Optional[LabelData]


@dataclass
class Sentence:
    tokens: list[Annotation]
