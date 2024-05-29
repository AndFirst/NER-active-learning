from app.utils import create_unique_folder_name


def test_create_unique_folder_name_no_existing_folder(tmp_path):
    path = tmp_path / "projects"
    path.mkdir()
    project_name = "test_project"
    assert create_unique_folder_name(str(path), project_name) == project_name


def test_create_unique_folder_name_existing_folder_no_number(tmp_path):
    path = tmp_path / "projects"
    path.mkdir()
    project_name = "test_project"
    (path / project_name).mkdir()
    assert create_unique_folder_name(str(path), project_name) == f"{project_name}_1"


def test_create_unique_folder_name_existing_folder_with_number(tmp_path):
    path = tmp_path / "projects"
    path.mkdir()
    project_name = "test_project"
    (path / f"{project_name}").mkdir()
    (path / f"{project_name}_1").mkdir()
    assert create_unique_folder_name(str(path), project_name) == f"{project_name}_2"
