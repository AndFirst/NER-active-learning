from typing import Dict, List, Tuple

from app.learning.file_wrappers.labeled.wrapper import LabeledWrapper
from app.learning.file_wrappers.unlabeled.wrapper import UnlabeledWrapper


class Dataset:
    def __init__(
        self,
        labeled_file: LabeledWrapper,
        unlabeled_file: UnlabeledWrapper,
        labels_to_idx: Dict[str, int],
        words_to_idx: Dict[str, int],
        padding_label: str,
        padding_idx: int,
        unlabeled_label: str,
        unlabeled_idx: int,
    ):
        self._labeled_file = labeled_file
        self._unlabeled_file = unlabeled_file

        self._labels_to_idx: Dict[str, int] = labels_to_idx
        self._words_to_idx: Dict[str, int] = words_to_idx

        self._idx_to_words: Dict[int, str] = {
            v: k for k, v in words_to_idx.items()
        }
        self._idx_to_labels: Dict[int, str] = {
            v: k for k, v in labels_to_idx.items()
        }

        self._padding_label: str = padding_label
        self._padding_idx: int = padding_idx

        self._unlabeled_label: str = unlabeled_label
        self._unlabeled_idx: int = unlabeled_idx

        self._longest_sentence_length = max(
            len(self._labeled_file.get_longest_sentence()),
            len(self._unlabeled_file.get_longest_sentence()),
        )

    def get_unlabeled_sentence(self, idx: int) -> List[str]:
        return self._unlabeled_file.get_sentence(idx)

    def _extract_features_and_labels(
        self, sentence: List[str]
    ) -> Tuple[List[int], List[int]]:
        assert len(sentence) % 2 == 0
        middle = len(sentence) // 2
        words, labels = sentence[:middle], sentence[middle:]
        return [self._words_to_idx[word] for word in words], [
            self._labels_to_idx[label] for label in labels
        ]

    def map_unlabeled_sentence_to_indices(
        self, sentence: List[str]
    ) -> List[int]:
        return [self._words_to_idx[word] for word in sentence]

    def map_indices_to_labels(self, indices: List[int]) -> List[str]:
        return [self._idx_to_labels[idx] for idx in indices]

    def move_sentence_to_labeled(self, sentence: List[str]) -> None:
        assert len(sentence) % 2 == 0
        middle = len(sentence) // 2
        words = sentence[:middle]
        idx = self._unlabeled_file.get_sentence_idx(words)
        self._labeled_file.save_sentence(sentence)
        self._unlabeled_file.remove_sentence(idx)

    def _apply_padding(self, vector: List[int]) -> List[int]:
        return vector[: self._longest_sentence_length] + [
            self._padding_idx
        ] * (self._longest_sentence_length - len(vector))

    def get_training_data(self) -> Tuple[List[List[int]], List[List[int]]]:
        sentences = self._labeled_file.get_all_sentences()

        features, labels = zip(
            *[
                self._extract_features_and_labels(sentence)
                for sentence in sentences
            ]
        )
        features = [self._apply_padding(feature) for feature in features]
        labels = [self._apply_padding(label) for label in labels]
        return features, labels

    def save(self):
        self._labeled_file.save()
        self._unlabeled_file.save()
