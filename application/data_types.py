from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class LabelData:
    label: str
    idx: int
    color: Tuple[int, int, int, int]


@dataclass
class AnnotateLabel:
    word: str
    label: Optional[LabelData]
