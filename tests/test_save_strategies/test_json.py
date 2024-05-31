import pytest
from unittest.mock import patch, mock_open
from app.learning.data_persistence_strategies.json import (
    JsonListStrategy,
    Sentences,
)


def test_json_list_strategy_init():
    strategy = JsonListStrategy("test.json")
    assert strategy._file_path == "test.json"


def test_json_list_strategy_init_invalid_file_type():
    with pytest.raises(ValueError):
        JsonListStrategy("test.txt")


@patch("os.path.isfile", return_value=True)
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='[["sentence1", "sentence2"]]',
)
def test_json_list_strategy_load(mock_open, mock_isfile):
    strategy = JsonListStrategy("test.json")
    result = strategy.load()
    assert result == [["sentence1", "sentence2"]]


@patch("os.path.isfile", return_value=False)
@patch("builtins.open", new_callable=mock_open)
def test_json_list_strategy_load_creates_file_if_not_exists(mock_open, mock_isfile):
    strategy = JsonListStrategy("test.json")
    result = strategy.load()
    assert result == []
    mock_open().write.assert_called_once_with("[]")


@patch("os.path.isfile", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='["incorrect_format"]')
def test_json_list_strategy_load_incorrect_format(mock_open, mock_isfile):
    strategy = JsonListStrategy("test.json")
    with pytest.raises(ValueError, match="JSON format is incorrect"):
        strategy.load()


@patch("os.path.isfile", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data="not_a_json")
def test_json_list_strategy_load_json_decode_error(mock_open, mock_isfile):
    strategy = JsonListStrategy("test.json")
    result = strategy.load()
    assert result == []


@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
def test_json_list_strategy_save(mock_json_dump, mock_open):
    strategy = JsonListStrategy("test.json")
    sentences: Sentences = [["sentence1", "sentence2"]]
    strategy.save(sentences)
    mock_json_dump.assert_called_once_with(sentences, mock_open(), ensure_ascii=False, indent=4)
