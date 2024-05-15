from typing import Dict, Tuple

from app.constants import DEFAULT_UNLABELED_LABEL
from app.learning.dataset.dataset import Dataset
from app.learning.models.ner_model import NERModel
from app.data_types import Sentence, Annotation, Word, LabelData


class ActiveLearningManager:
    def __init__(
        self,
        model: NERModel,
        dataset: Dataset,
        labels: Dict[str, Tuple[int, int, int, int]],
        batch_size: int = 1,
        epochs: int = 20,
    ) -> None:
        self._dataset = dataset
        self._batch_size: int = batch_size
        self._epochs: int = epochs
        self._model: NERModel = model

        self._label_mapping = labels

    def _get_sentence_idx(self) -> int:
        return 0

    def get_sentence(self, annotated=False) -> Sentence | None:
        idx = self._get_sentence_idx()
        sentence = self._dataset.get_unlabeled_sentence(idx)

        if not annotated:
            annotations = [
                Annotation(words=[Word(word)], label=None) for word in sentence
            ]
            return Sentence(annotations)
        else:
            word_indices = self._dataset.map_unlabeled_sentence_to_indices(
                sentence
            )

            labels_indices = self._model.predict(word_indices)

            labels = self._dataset.map_indices_to_labels(labels_indices)

            annotations = []
            for i, label in enumerate(labels):
                if label == DEFAULT_UNLABELED_LABEL:
                    annotations.append(
                        Annotation(words=[Word(sentence[i])], label=None)
                    )
                elif (
                    label[0:2] == "I-"
                    and i != 0
                    and labels[i - 1][0:2] == "B-"
                    and label[2:] == labels[i - 1][2:]
                ):
                    annotations[-1].words.append(Word(sentence[i]))
                else:
                    label_name = label[2:]
                    label_data = LabelData(
                        label=label_name,
                        color=self._label_mapping[label_name],
                    )
                    annotations.append(
                        Annotation(words=[Word(sentence[i])], label=label_data)
                    )
            return Sentence(annotations)

    def give_feedback(self, sentence: Sentence) -> None:
        converted_sentence = sentence.to_list()
        self._dataset.move_sentence_to_labeled(converted_sentence)

        features, target = self._dataset.get_training_data()

        self._model.train(
            features, target, epochs=self._epochs, batch_size=self._batch_size
        )

    @property
    def label_mapping(self):
        return self._label_mapping
