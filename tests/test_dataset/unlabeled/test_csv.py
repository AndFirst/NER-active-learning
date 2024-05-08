from unittest.mock import patch, mock_open

import pytest

from app.learning.file_wrappers.unlabeled.csv import UnlabeledCsv


def test_create_csv_unlabeled_wrapper():
    wrapper = UnlabeledCsv("project/unlabeled.csv")
    assert wrapper._file_path == "project/unlabeled.csv"


def test_create_csv_unlabeled_wrapper_invalid_file_type():
    with pytest.raises(ValueError):
        wrapper = UnlabeledCsv("project/unlabeled.json")


@patch("builtins.open", new_callable=mock_open, read_data="Word1\tWord2\nWord3\tWord4\nWord5\tWord6")
def test_count_sentences(mock_file_open):
    with patch("os.path") as mock_path:
        mock_path.isfile.return_value = True
        wrapper = UnlabeledCsv("project/unlabeled.csv")
        assert wrapper.count_sentences() == 3


@patch("builtins.open", new_callable=mock_open, read_data="Word1\tWord2\nWord3\tWord4\nWord5\tWord6")
def test_get_sentence(mock_file_open):
    with patch("os.path") as mock_path:
        mock_path.isfile.return_value = True
        wrapper = UnlabeledCsv("project/unlabeled.csv")
        assert wrapper.get_sentence(0) == ["Word1", "Word2"]
        assert wrapper.get_sentence(1) == ["Word3", "Word4"]
        assert wrapper.get_sentence(2) == ["Word5", "Word6"]


@patch("builtins.open", new_callable=mock_open, read_data="Word1\tWord2\nWord3\tWord4\nWord5\tWord6")
def test_get_sentence_with_invalid_index(mock_file_open):
    with patch("os.path") as mock_path:
        mock_path.isfile.return_value = True
        wrapper = UnlabeledCsv("project/unlabeled.csv")
        with pytest.raises(IndexError):
            wrapper.get_sentence(5)


@patch("builtins.open", new_callable=mock_open, read_data="Word1\tWord2\nWord1\tWord2\nWord3\tWord1")
def test_unique_words(mock_file_open):
    with patch("os.path") as mock_path:
        mock_path.isfile.return_value = True
        wrapper = UnlabeledCsv("project/unlabeled.csv")
        unique_words = wrapper.unique_words()
        assert unique_words == {"Word1", "Word2", "Word3"}
