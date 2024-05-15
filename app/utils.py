import torch
import os


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


def create_model(data):
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
