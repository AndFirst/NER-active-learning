import csv
import json
import threading
from typing import List, Tuple

from app.data_preparation import get_longest_sentence_length_from_dataset
from app.learning.models.ner_model import NERModel
from app.file_operations import remove_sentence_from_csv, count_csv_rows
from app.data_types import Sentence, Annotation, Word, LabelData


class ActiveLearningManager:
    def __init__(
        self,
        labeled_path: str,
        unlabeled_path: str,
        word_to_idx_path: str,
        label_to_idx_path: str,
        label_mapping: str,
        model: NERModel,
    ) -> None:
        self._unlabeled_path = unlabeled_path
        self._labeled_path = labeled_path
        self._current_sentence_idx: int | None = None
        self._batch_size: int = 1

        self._model: NERModel = model

        self._label_mapping = label_mapping

        with open(word_to_idx_path) as file:
            self._word_to_idx = json.load(file)
        self._idx_to_word = {v: k for k, v in self._word_to_idx.items()}

        with open(label_to_idx_path) as file:
            self._label_to_idx = json.load(file)
        self._idx_to_label = {v: k for k, v in self._label_to_idx.items()}

        with open(self._unlabeled_path, "r") as file:
            reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
            dataset = list(reader)
            self._max_sentence_length = (
                get_longest_sentence_length_from_dataset(dataset)
            )
            self._max_sentence_length = 50

    def get_sentence(self, annotated=False) -> Sentence | None:
        with open(self._unlabeled_path, "r") as file:
            reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
            row = next(reader, None)
        if row is not None:
            if not annotated:
                annotations = [
                    Annotation(words=[Word(word)], label=None) for word in row
                ]
                return Sentence(annotations)
            else:
                # words to idx
                words_indexes = [self._word_to_idx[word] for word in row]

                # model predict
                labels_indexes = self._model.predict(words_indexes)

                # labels to labeled sentence
                labels = [self._idx_to_label[idx] for idx in labels_indexes]
                annotations = []
                for i, label in enumerate(labels):
                    if label == "_":
                        annotations.append(
                            Annotation(words=[Word(row[i])], label=None)
                        )
                    elif (
                        label[0:2] == "I-"
                        and i != 0
                        and labels[i - 1][0:2] == "B-"
                        and label[2:] == labels[i - 1][2:]
                    ):
                        annotations[-1].words.append(Word(row[i]))
                    else:
                        label_name = label[2:]
                        label_data = LabelData(
                            label=label_name,
                            color=self._label_mapping[label_name],
                        )
                        annotations.append(
                            Annotation(words=[Word(row[i])], label=label_data)
                        )
                return Sentence(annotations)
        else:
            return None

    def train_model_async(self, features, labels, epochs, batch_size):
        thread = threading.Thread(
            target=self._model.train,
            args=(features, labels, epochs, batch_size),
        )
        thread.start()

    def give_feedback(self, sentence: Sentence) -> None:
        with open(self._labeled_path, "a") as file:
            sentence.to_csv(file)

        with open(
            self._unlabeled_path,
            "r+",
        ) as file:
            remove_sentence_from_csv(0, file, file)

        with open(self._labeled_path, "r") as file:
            num_rows = count_csv_rows(file)

        if num_rows % self._batch_size == 0:
            with open(self._labeled_path, "r") as file:
                reader = csv.reader(
                    file, delimiter="\t", quoting=csv.QUOTE_NONE
                )
                sentences = list(reader)
            last_sentences = sentences[-self._batch_size :]
            last_features, last_labels = self.extract_features_and_targets(
                last_sentences
            )
            last_features = [
                [self._word_to_idx[word] for word in sentence]
                for sentence in last_features
            ]
            last_labels = [
                [self._label_to_idx[label] for label in labels]
                for labels in last_labels
            ]
            last_features = [
                self.add_padding_to_vector(feature)
                for feature in last_features
            ]
            last_labels = [
                self.add_padding_to_vector(labels) for labels in last_labels
            ]

            old_sentences = sentences[: -self._batch_size]
            old_features, old_labels = self.extract_features_and_targets(
                old_sentences
            )

            old_features = [
                [self._word_to_idx[word] for word in sentence]
                for sentence in old_features
            ]
            old_labels = [
                [self._label_to_idx[label] for label in labels]
                for labels in old_labels
            ]

            old_features = [
                self.add_padding_to_vector(feature) for feature in old_features
            ]
            old_labels = [
                self.add_padding_to_vector(labels) for labels in old_labels
            ]

            # TODO can be train on full dataset or only on new data idk
            self._model.train(
                last_features, last_labels, 20, batch_size=self._batch_size
            )
            if old_sentences:
                self._model.train(
                    old_features, old_labels, 20, batch_size=self._batch_size
                )

    def add_padding_to_vector(
        self, vector: List[int], padding_idx: int = 0
    ) -> List[int]:
        return vector[: self._max_sentence_length] + [padding_idx] * (
            self._max_sentence_length - len(vector)
        )

    def cut_sentence_padding(
        self, sentence: List[int], labels: List[int], padding_idx: int = 35178
    ) -> Tuple[List[int], List[int]]:
        if padding_idx not in sentence:
            return sentence, labels

        idx_of_padding = sentence.index(padding_idx)
        return sentence[:idx_of_padding], labels[:idx_of_padding]

    @staticmethod
    def extract_features_and_targets(
        sentences: List[List[str]],
    ) -> Tuple[List[List[str]], List[List[str]]]:
        features = []
        targets = []
        for sentence in sentences:
            end_sentence_index = sentence.index("<END_SENTENCE>")
            features.append(sentence[:end_sentence_index])
            targets.append(sentence[end_sentence_index + 1 :])
        return features, targets
