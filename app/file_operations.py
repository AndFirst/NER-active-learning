import csv
import json
from typing import Dict, Hashable, Any, List, IO

from data_preparation import (
    labels_to_numbers,
    words_to_numbers,
    human_readable_to_model_labels,
)
from data_types import ProjectData, Word, LabelData
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


def generateModel(path, model_type, num_cls):
    import torch
    match model_type:
        case "BiLSTM":
            from ..models import BiLSTM
            model = BiLSTM(num_classes=num_cls)
            torch.save(model, path)
        case _:
            pass


def save_project(project_data: ProjectData):
    directory_path = project_data.save_path
    os.makedirs(directory_path)
    _, extension = os.path.splitext(project_data.dataset_path)

    unlabeled_path = os.path.join(directory_path, "unlabeled" + extension)
    labeled_path = os.path.join(directory_path, "labeled" + extension)
    project_path = os.path.join(directory_path, "project.json")
    word_to_vec_path = os.path.join(directory_path, "word_to_vec.json")
    label_to_vec_path = os.path.join(directory_path, "label_to_vec.json")
    model_path = os.path.join(directory_path, "model.pth")

    shutil.copy(project_data.dataset_path, unlabeled_path)

    with open(project_path, "w") as json_file:
        json.dump(project_data.to_dict(), json_file)

    with open(labeled_path, "w"):
        # create empty file
        pass

    with open(unlabeled_path, "r") as file:
        words = get_words_from_csv(file)

    with open(label_to_vec_path, "w") as file:
        labels = [LabelData(**label) for label in project_data.labels]
        labels = human_readable_to_model_labels(labels)
        label_to_vec = labels_to_numbers(labels)
        json.dump(label_to_vec, file)

    with open(word_to_vec_path, "w") as file:
        word_to_vec = words_to_numbers(words)
        json.dump(word_to_vec, file)

    generateModel(model_path, project_data.model, len(project_data.labels))


def get_words_from_csv(fh: IO) -> List[Word]:
    reader = csv.reader(fh, delimiter="\t")
    rows_list = [Word(item) for row in reader for item in row]
    return rows_list


def remove_sentence_from_csv(file_name, sentence_to_remove):
    lines = []
    words = sentence_to_remove.split("\t")
    with open(file_name, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                text = row[0].split("\t")
                print(text)
                print(words)
                print("======================")
                if words != text:
                    print("chuj")
                    lines.append(row)
                else:
                    print("duppaaa")
    print(len(lines))
    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(lines)


def load_project_from_file(file_path: str) -> dict:
    with open(file_path, "r") as json_file:
        project_data = json.load(json_file)
    return project_data


def save_to_json(data_to_save: Dict[Hashable, Any], file_path: str) -> None:
    with open(file_path, "w") as json_file:
        json.dump(data_to_save, json_file)


if __name__ == "__main__":
    with open(
        "/home/irek/PycharmProjects/zprp-ner-active_learning/data/processed/unlabeled.csv",
        "r",
    ) as csv_file:
        get_words_from_csv(csv_file)
