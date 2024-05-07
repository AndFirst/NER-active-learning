from typing import List
from models.ner_model import NERModel


class Dummy(NERModel):
    def __init__(self):
        super().__init__()

    def predict(self, unlabeled_sentence: List[int]) -> List[int]:
        import random

        return [random.randint(0, 6) for sentence in unlabeled_sentence]

    def train(
        self,
        features: List[List[int]],
        targets: List[List[int]],
        epochs: int,
        batch_size: int,
    ) -> None: ...

    def reset(self) -> None: ...

    @classmethod
    def from_torch(cls, file_path: str) -> NERModel:
        raise NotImplementedError
