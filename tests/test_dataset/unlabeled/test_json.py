from unittest.mock import patch, mock_open

import pytest
from app.learning.file_wrappers.unlabeled.json import UnlabeledJson


def test_create_json_unlabeled_wrapper():
    wrapper = UnlabeledJson("project/unlabeled.json")
    assert wrapper._file_path == "project/unlabeled.json"


def test_create_json_unlabeled_wrapper_invalid_file_type():
    with pytest.raises(ValueError):
        UnlabeledJson("project/unlabeled.csv")


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='[["Word1", "Word2"], ["Word3", "Word4"], ["Word5", "Word6"]]',
)
def test_count_sentences(mock_file_open):
    with patch("os.path") as mock_path:
        mock_path.isfile.return_value = True
        wrapper = UnlabeledJson("project/unlabeled.json")
        assert wrapper.count_sentences() == 3


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='[["Word1", "Word2"], ["Word3", "Word4"], ["Word5", "Word6"]]',
)
def test_get_sentence(mock_file_open):
    with patch("os.path") as mock_path:
        mock_path.isfile.return_value = True
        wrapper = UnlabeledJson("project/unlabeled.json")
        assert wrapper.get_sentence(0) == ["Word1", "Word2"]
        assert wrapper.get_sentence(1) == ["Word3", "Word4"]
        assert wrapper.get_sentence(2) == ["Word5", "Word6"]


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='[["Word1", "Word2"], ["Word3", "Word4"], ["Word5", "Word6"]]',
)
def test_get_sentence_with_invalid_index(mock_file_open):
    with patch("os.path") as mock_path:
        mock_path.isfile.return_value = True
        wrapper = UnlabeledJson("project/unlabeled.json")
        with pytest.raises(IndexError):
            wrapper.get_sentence(3)


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='[["Word1", "Word2"], ["Word2", "Word3"], ["Word3", "Word4"], ["Word5", "Word6"]]',
)
def test_unique_words(mock_file_open):
    with patch("os.path") as mock_path:
        mock_path.isfile.return_value = True
        wrapper = UnlabeledJson("project/unlabeled.json")
        unique_words = wrapper.unique_words()
        assert unique_words == {
            "Word1",
            "Word2",
            "Word3",
            "Word4",
            "Word5",
            "Word6",
        }
