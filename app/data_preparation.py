from typing import List, Dict

from data_types import LabelData


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


def words_to_numbers(words: List[str]) -> Dict[str, int]:
    return {word: idx for idx, word in enumerate(words)}


def get_unique_words_from_dataset(dataset: List[str]) -> List[str]:
    return list(set(dataset))


def get_longest_sentence_length_from_dataset(dataset: List[List[str]]) -> int:
    if not dataset:
        return 0

    longest_sentence_length = max(len(sentence) for sentence in dataset)
    return longest_sentence_length
