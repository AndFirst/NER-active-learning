from typing import List, Dict

from app.data_types import LabelData, Word


def human_readable_to_model_labels(labels: List[LabelData]) -> List[LabelData]:
    prefixes = "B-", "I-"
    empty_label = "_"
    labels = [LabelData(label=empty_label, color=(0, 0, 0, 0))] + [
        LabelData(label=prefix + label.label, color=label.color)
        for label in labels
        for prefix in prefixes
    ]
    return labels


def labels_to_numbers(labels: List[LabelData]) -> Dict[str, int]:
    return {label.label: idx for idx, label in enumerate(labels)}


def words_to_numbers(words: List[Word]) -> Dict[str, int]:
    return {word.word: idx for idx, word in enumerate(words)}


def get_unique_words_from_dataset(dataset: List[str]) -> List[str]:
    return list(set(dataset))
