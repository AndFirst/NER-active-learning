import json

from data_types import ProjectData
import os
import shutil


def create_unique_folder_name(directory: str, project_name: str) -> str:
    if os.path.exists(os.path.join(directory, project_name)):
        subfolders = [
            f
            for f in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, f))
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


def save_project(project_data: ProjectData, directory: str):
    project_name = "_".join(project_data.name.split())
    if not os.path.exists(directory):
        os.makedirs(directory)

    unique_folder_name = create_unique_folder_name(directory, project_name)
    new_folder_path = os.path.join(directory, unique_folder_name)
    os.makedirs(new_folder_path)

    # Extracting extension from the original file path
    _, extension = os.path.splitext(project_data.dataset_path)

    # Copying the file to the new folder with the same extension and name "unlabelled"
    new_file_path = os.path.join(new_folder_path, "unlabeled" + extension)
    shutil.copy(project_data.dataset_path, new_file_path)
    # Saving project data to JSON file
    with open(os.path.join(new_folder_path, "project.json"), "w") as json_file:
        json.dump(project_data.to_dict(), json_file)
