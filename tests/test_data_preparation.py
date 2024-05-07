# from app.data_types import LabelData, Word
# from app.data_preparation import (
#     human_readable_to_model_labels,
#     labels_to_numbers,
#     get_unique_words_from_dataset,
#     words_to_numbers,
# )
#
#
# def test_empty_labels():
#     labels = []
#     result = human_readable_to_model_labels(labels)
#     assert result == [LabelData(label="_", color=(0, 0, 0, 0))]
#
#
# def test_single_label():
#     labels = [LabelData(label="entity", color=(0, 0, 255, 255))]
#     result = human_readable_to_model_labels(labels)
#     expected_result = [
#         LabelData(label="_", color=(0, 0, 0, 0)),
#         LabelData(label="B-entity", color=(0, 0, 255, 255)),
#         LabelData(label="I-entity", color=(0, 0, 255, 255)),
#     ]
#     assert result == expected_result
#
#
# def test_multiple_labels():
#     labels = [
#         LabelData(label="entity", color=(0, 0, 255, 255)),
#         LabelData(label="action", color=(255, 0, 0, 255)),
#     ]
#     result = human_readable_to_model_labels(labels)
#     expected_result = [
#         LabelData(label="_", color=(0, 0, 0, 0)),
#         LabelData(label="B-entity", color=(0, 0, 255, 255)),
#         LabelData(label="I-entity", color=(0, 0, 255, 255)),
#         LabelData(label="B-action", color=(255, 0, 0, 255)),
#         LabelData(label="I-action", color=(255, 0, 0, 255)),
#     ]
#     assert result == expected_result
#
#
# def test_empty_labels_to_numbers():
#     labels = []
#     result = labels_to_numbers(labels)
#     assert result == {}
#
#
# def test_single_label_to_numbers():
#     labels = [LabelData(label="entity", color=(0, 0, 255, 255))]
#     result = labels_to_numbers(labels)
#     expected_result = {LabelData(label="entity", color=(0, 0, 255, 255)): 0}
#     assert result == expected_result
#
#
# def test_multiple_labels_to_numbers():
#     labels = [
#         LabelData(label="entity", color=(0, 0, 255, 255)),
#         LabelData(label="action", color=(255, 0, 0, 255)),
#         LabelData(label="attribute", color=(0, 255, 0, 255)),
#     ]
#     result = labels_to_numbers(labels)
#     expected_result = {
#         LabelData(label="entity", color=(0, 0, 255, 255)): 0,
#         LabelData(label="action", color=(255, 0, 0, 255)): 1,
#         LabelData(label="attribute", color=(0, 255, 0, 255)): 2,
#     }
#     assert result == expected_result
#
#
# def test_empty_words_to_numbers():
#     words = []
#     result = words_to_numbers(words)
#     assert result == {}
#
#
# def test_single_word_to_numbers():
#     word_entity = Word(word="entity")
#     words = [word_entity]
#     result = words_to_numbers(words)
#     expected_result = {word_entity: 0}
#     assert result == expected_result
#
#
# def test_multiple_words_to_numbers():
#     word_entity = Word(word="entity")
#     word_action = Word(word="action")
#     word_attribute = Word(word="attribute")
#     words = [word_entity, word_action, word_attribute]
#     result = words_to_numbers(words)
#     expected_result = {word_entity: 0, word_action: 1, word_attribute: 2}
#     assert result == expected_result
#
#
# def test_get_unique_words_from_dataset():
#     dataset = [
#         "Thousands",
#         "of",
#         "demonstrators",
#         "have",
#         "marched",
#         "through",
#         "London",
#         "to",
#         "protest",
#         "the",
#         "war",
#         "in",
#         "Iraq",
#         "and",
#         "demand",
#         "withdrawal",
#         "of",
#         "British",
#         "troops",
#         "from",
#         "that",
#         "country",
#         ".",
#         "Iranian",
#         "officials",
#         "say",
#         "they",
#         "expect",
#         "to",
#         "get",
#         "access",
#         "to",
#         "sealed",
#         "sensitive",
#         "parts",
#         "of",
#         "the",
#         "plant",
#         "Wednesday",
#         ",",
#         "after",
#         "an",
#         "IAEA",
#         "surveillance",
#         "system",
#         "begins",
#         "functioning",
#         ".",
#     ]
#
#     result = get_unique_words_from_dataset(dataset)
#     expected_result = [
#         "Thousands",
#         "of",
#         "demonstrators",
#         "have",
#         "marched",
#         "through",
#         "London",
#         "to",
#         "protest",
#         "the",
#         "war",
#         "in",
#         "Iraq",
#         "and",
#         "demand",
#         "withdrawal",
#         "British",
#         "troops",
#         "from",
#         "that",
#         "country",
#         ".",
#         "Iranian",
#         "officials",
#         "say",
#         "they",
#         "expect",
#         "get",
#         "access",
#         "sealed",
#         "sensitive",
#         "parts",
#         "plant",
#         "Wednesday",
#         ",",
#         "after",
#         "an",
#         "IAEA",
#         "surveillance",
#         "system",
#         "begins",
#         "functioning",
#     ]
#
#     assert set(result) == set(expected_result)
