import csv
import json
from typing import Dict, Hashable, Any, List, IO

import torch
from app.data_preparation import (
    labels_to_numbers,
    words_to_numbers,
    human_readable_to_model_labels,
    get_unique_words_from_dataset,
)
from app.data_types import ProjectData, Word, LabelData
import os
import shutil


def create_unique_folder_name(path: str, project_name: str) -> str:
    if os.path.exists(os.path.join(path, project_name)):
        subfolders = [
            f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))
        ]
        existing_numbers = [
            int(folder.split("_")[-1])
            for folder in subfolders
            if folder.startswith(project_name + "_")
        ]
        if existing_numbers:
            return f"{project_name}_{max(existing_numbers) + 1}"
        else:
            return f"{project_name}_1"
    else:
        return project_name


def save_project(project_data: ProjectData):
    directory_path = project_data.save_path
    os.makedirs(directory_path)
    _, extension = os.path.splitext(project_data.dataset_path)

    unlabeled_path = os.path.join(directory_path, "unlabeled" + extension)
    labeled_path = os.path.join(directory_path, "labeled" + extension)
    project_path = os.path.join(directory_path, "project.json")
    word_to_vec_path = os.path.join(directory_path, "word_to_vec.json")
    label_to_vec_path = os.path.join(directory_path, "label_to_vec.json")
    os.path.join(directory_path, "model.pth")

    shutil.copy(project_data.dataset_path, unlabeled_path)

    with open(project_path, "w") as json_file:
        json.dump(project_data.to_dict(), json_file)

    with open(labeled_path, "w"):
        # create empty file
        pass

    with open(unlabeled_path, "r") as file:
        words = get_words_from_csv(file)
        words = get_unique_words_from_dataset(words)

    with open(label_to_vec_path, "w") as file:
        labels = [LabelData(**label) for label in project_data.labels]
        labels = human_readable_to_model_labels(labels)
        label_to_vec = labels_to_numbers(labels)
        json.dump(label_to_vec, file)

    with open(word_to_vec_path, "w") as file:
        word_to_vec = words_to_numbers(words)
        json.dump(word_to_vec, file)


def get_words_from_csv(fh: IO) -> List[str]:
    reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
    rows_list = [item for row in reader for item in row]
def create_model(data):
    print(data)
    model_type = data["model"]
    path = data[""]
    match model_type:
        case "BiLSTM":
            from ..models import BiLSTM

            model = BiLSTM(num_classes=1)
            torch.save(model, path)
        case "Your model":
            data["user_model_path"]
            # copy model to folder\model.pth
        case _:
            pass


def get_words_from_csv(fh: IO) -> List[Word]:
    reader = csv.reader(fh, delimiter="\t")
    rows_list = [Word(item) for row in reader for item in row]
    return rows_list


def remove_sentence_from_csv(
    sentence_idx: int, input_file: IO, output_file: IO
):
    csv_reader = csv.reader(input_file, delimiter="\t", quoting=csv.QUOTE_NONE)
    rows = list(csv_reader)

    if 0 <= sentence_idx < len(rows):
        del rows[sentence_idx]
    else:
        raise ValueError(f"Index {sentence_idx} out of range")

    input_file.seek(0)
    output_file.truncate(0)
    for row in rows:
        output_file.write("\t".join(row) + "\n")


def load_project_from_file(file_path: str) -> dict:
    with open(file_path, "r") as json_file:
        project_data = json.load(json_file)
    return project_data


def save_to_json(data_to_save: Dict[Hashable, Any], file_path: str) -> None:
    with open(file_path, "w") as json_file:
        json.dump(data_to_save, json_file)


def count_csv_rows(fh: IO) -> int:
    reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
    return len(list(reader))
