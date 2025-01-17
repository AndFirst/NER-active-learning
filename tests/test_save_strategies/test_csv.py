import pytest
from unittest.mock import patch, mock_open
from app.learning.data_persistence_strategies.csv import (
    CsvTabSeparatedStrategy,
    Sentences,
)


def test_csv_tab_separated_strategy_init():
    strategy = CsvTabSeparatedStrategy("test.csv")
    assert strategy._file_path == "test.csv"


def test_csv_tab_separated_strategy_init_invalid_file_type():
    with pytest.raises(ValueError):
        CsvTabSeparatedStrategy("test.txt")


@patch("builtins.open", new_callable=mock_open, read_data="sentence1\tsentence2\n")
@patch("os.path.isfile", return_value=True)
def test_csv_tab_separated_strategy_load(mock_isfile, mock_file):
    strategy = CsvTabSeparatedStrategy("test.csv")
    result = strategy.load()
    assert result == [["sentence1", "sentence2"]]


@patch("builtins.open", new_callable=mock_open)
def test_csv_tab_separated_strategy_save(mock_file):
    strategy = CsvTabSeparatedStrategy("test.csv")
    sentences: Sentences = [["sentence1", "sentence2"]]
    strategy.save(sentences)
    mock_file.assert_called_once_with("test.csv", "w")
    mock_file().write.assert_called_once_with("sentence1\tsentence2\n")
