import json
from data_types import ProjectData
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
    project_name = "_".join(project_data.name.split())
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # print(directory)
    path = project_data.save_path
    unique_folder_name = create_unique_folder_name(path, project_name)
    new_folder_path = os.path.join(path, unique_folder_name)
    os.makedirs(new_folder_path)
    print(unique_folder_name)
    _, extension = os.path.splitext(project_data.dataset_path)
    new_file_path = os.path.join(new_folder_path, "unlabeled" + extension)
    shutil.copy(project_data.dataset_path, new_file_path)
    with open(os.path.join(new_folder_path, "project.json"), "w") as json_file:
        json.dump(project_data.to_dict(), json_file)
    with open(os.path.join(new_folder_path, "labeled" + extension), "w"):
        pass


def load_project_from_file(file_path: str) -> dict:
    with open(file_path, "r") as json_file:
        project_data = json.load(json_file)
    return project_data


if __name__ == "__main__":
    project = ProjectData(
        name="dupa",
        description="dupa project",
        save_path="/home/irek/PycharmProjects/zprp-ner-active_learning/app/saved_projects",
        dataset_path="/home/irek/PycharmProjects/zprp-ner-active_learning/app/datasets/dataset.csv",
    )
    save_project(project)
