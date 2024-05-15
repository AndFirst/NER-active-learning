from typing import List

from app.learning.models.ner_model import NERModel


class CustomModel(NERModel):
    def train(
        self,
        features: List[List[int]],
        targets: List[List[int]],
        epochs: int,
        batch_size: int,
    ) -> None:
        pass

    def predict(self, unlabeled_sentence: List[int]) -> List[int]:
        pass

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, file_path: str) -> NERModel:
        pass

    ...
