from app.data_types import LabelData


def test_is_empty():
    assert LabelData(label="", color=(0, 0, 0, 0)).is_empty() is True
    assert LabelData(label=" ", color=(0, 0, 0, 0)).is_empty() is True
    assert LabelData(label="label", color=(0, 0, 0, 0)).is_empty() is False


def test_equality():
    label1 = LabelData(label="Label", color=(255, 0, 0, 255))
    label2 = LabelData(label="label", color=(0, 255, 0, 255))
    label3 = LabelData(label="OtherLabel", color=(0, 0, 255, 255))

    assert label1 == label2
    assert label1 != label3


def test_hash():
    label1 = LabelData(label="Label", color=(255, 0, 0, 255))
    label2 = LabelData(label="label", color=(0, 255, 0, 255))
    label3 = LabelData(label="OtherLabel", color=(0, 0, 255, 255))

    assert hash(label1) == hash(label2)
    assert hash(label1) != hash(label3)


def test_to_dict():
    label = LabelData(label="Label", color=(255, 0, 0, 255))
    expected_dict = {"label": "Label", "color": (255, 0, 0, 255)}

    assert label.to_dict() == expected_dict


def test_invalid_equality():
    label = LabelData(label="Label", color=(255, 0, 0, 255))
    not_a_label = "NotALabel"

    assert label != not_a_label
