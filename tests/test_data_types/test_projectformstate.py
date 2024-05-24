from app.data_types import ProjectFormState, LabelData


def test_input_extension():
    project_form_state = ProjectFormState(dataset_path="data/dataset.csv")
    assert project_form_state.input_extension == ".csv"

    project_form_state.dataset_path = "data/dataset.txt"
    assert project_form_state.input_extension == ".txt"

    project_form_state.dataset_path = "data/dataset"
    assert project_form_state.input_extension == ""


def test_to_dict():
    label1 = LabelData(label="Label1", color=(255, 0, 0, 255))
    label2 = LabelData(label="Label2", color=(0, 255, 0, 255))
    project_form_state = ProjectFormState(
        name="TestProject",
        description="TestDescription",
        save_path="/path/to/save",
        dataset_path="/path/to/dataset.csv",
        labels=[label1, label2],
        model_type="TestModel",
        model_state_path="/path/to/model/state",
        model_implementation_path="/path/to/model/implementation",
        output_extension=".out",
    )

    expected_dict = {
        "name": "TestProject",
        "model_type": "TestModel",
        "description": "TestDescription",
        "save_path": "/path/to/save",
        "dataset_path": "/path/to/dataset.csv",
        "labels": [label1.to_dict(), label2.to_dict()],
        "model_state_path": "/path/to/model/state",
        "model_implementation_path": "/path/to/model/implementation",
        "input_extension": ".csv",
        "output_extension": ".out",
    }

    assert project_form_state.to_dict() == expected_dict


def test_get_existing_property():
    project_form_state = ProjectFormState(name="TestProject")
    assert project_form_state.get("name", "DefaultName") == "TestProject"
    assert (
        project_form_state.get("nonexistent_property", "DefaultValue") == "DefaultValue"
    )


def test_get_default_property():
    project_form_state = ProjectFormState()
    assert (
        project_form_state.get("nonexistent_property", "DefaultValue") == "DefaultValue"
    )


def test_default_values():
    project_form_state = ProjectFormState()
    assert project_form_state.name == ""
    assert project_form_state.description == ""
    assert project_form_state.save_path == ""
    assert project_form_state.output_extension is None
    assert project_form_state.dataset_path == ""
    assert project_form_state.labels == []
    assert project_form_state.model_type == ""
    assert project_form_state.model_state_path is None
    assert project_form_state.model_implementation_path is None
