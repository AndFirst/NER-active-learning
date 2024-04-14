import json
from typing import List
import pandas as pd
import csv
from itertools import product


def create_unlabeled_dataset(
    path: str, json_path: str = None, csv_path: str = None
) -> None:
    """Create unlabeled dataset from sample dataset from Kaggle.
    https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus
    Delete all labels and merge words into sentences. Save data to choosen format.
    Supported formats: csv, json.

    Args:
        path (str): Path to original dataset.
        json_path (str, optional): Output json path. Defaults to None.
        csv_path (str, optional): Output csv path. Defaults to None.
    """
    assert (
        json_path is not None and csv_path is not None
    ), "Both paths cannot be None"
    data = pd.read_csv(path, encoding="latin1")
    data = data.fillna(method="ffill")

    data = (
        data.groupby("Sentence #")
        .apply(lambda s: s["Word"].values.tolist())
        .tolist()
    )
    if json_path:
        with open(json_path, "w", newline="") as file:
            json.dump(data, file)
    if csv_path:
        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerows(data)


def create_word_to_idx(path: str, output_path: str) -> None:
    """Generate json file with mapping words into indexes.
    Extract unique words from dataset. Add special word "<PAD>".
    Map all words to indices. Save to json.

    Args:
        path (str): Path to dataset.
        output_path (str): Path to output file.
    """
    words = []
    with open(path, "r", encoding="latin1") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            words.extend(row)
    words = list(set(words))
    word2idx = {word: idx + 1 for idx, word in enumerate(words)}
    word2idx["<PAD>"] = 0
    with open(output_path, "w") as file:
        json.dump(word2idx, file)


def create_tag_to_idx(tags: List[str], output_path: str) -> None:
    """Create labels for text annotating from given human readable labels.
    Add 'B-' and 'I-' for each given label. Create label for not-labeled words.
    Map all labels to indices. Save to json.

    Args:
        tags (List[str]): List of human-readable labels e.g. [Person, Geo, Animal, ...]
        output_path (str): Path to output file.
    """
    extended_tags = ["O"]
    extended_tags += ["".join(c) for c in product(("B-", "I-"), tags)]
    tag2idx = {tag: idx for idx, tag in enumerate(extended_tags)}
    with open(output_path, "w") as file:
        json.dump(tag2idx, file)
