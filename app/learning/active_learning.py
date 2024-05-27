from typing import Dict, Any

import numpy as np

from app.constants import DEFAULT_UNLABELED_LABEL
from app.learning.dataset import Dataset
from app.learning.models import NERModel
from app.data_types import Sentence, Annotation, Word, AssistantConf


class ActiveLearningManager:
    def __init__(
        self, model: NERModel, dataset: Dataset, config: AssistantConf
    ) -> None:
        self._dataset: Dataset = dataset
        self._config: AssistantConf = config
        self._model: NERModel = model
        self._annotated_sentences_count = 0

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

            labels_indices, labels_confidences = (
                self._model.predict_with_confidence(word_indices)
            )
            print(np.average(labels_confidences))

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
                    for label_ in self._config.labels:
                        if label_.label == label_name:
                            label_data = label_
                            break
                    annotations.append(
                        Annotation(words=[Word(sentence[i])], label=label_data)
                    )
            return Sentence(annotations)

    def give_feedback(self, sentence: Sentence) -> None:
        converted_sentence = sentence.to_list()
        self._dataset.move_sentence_to_labeled(converted_sentence)
        self._annotated_sentences_count += 1

        if self._annotated_sentences_count == self._config.sampling_batch_size:
            self._annotated_sentences_count = 0
            features, target = self._dataset.get_training_data()
            weights = self._dataset.get_weights()

            self._model.train_async(
                features,
                target,
                epochs=self._config.epochs,
                batch_size=self._config.batch_size,
                class_weights=weights,
            )

    @property
    def labels(self):
        return self._config.labels

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "labeled_sentences_count": self._dataset.labeled_sentences_count,
            "unlabeled_sentences_count": self._dataset.unlabeled_sentences_count,
        }
