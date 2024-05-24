from app.data_types import Word, Annotation, Sentence, LabelData


def test_get_left_neighbor():
    word1 = Word(word="word1")
    word2 = Word(word="word2")
    word3 = Word(word="word3")
    annotation1 = Annotation(
        words=[word1, word2], label=LabelData(label="Label1", color=(255, 0, 0, 255))
    )
    annotation2 = Annotation(
        words=[word3], label=LabelData(label="Label2", color=(0, 255, 0, 255))
    )
    sentence = Sentence(tokens=[annotation1, annotation2])

    assert sentence.get_left_neighbor(word1) is None
    assert sentence.get_left_neighbor(word2) == word1
    assert sentence.get_left_neighbor(word3) == word2


def test_get_right_neighbor():
    word1 = Word(word="word1")
    word2 = Word(word="word2")
    word3 = Word(word="word3")
    annotation1 = Annotation(
        words=[word1, word2], label=LabelData(label="Label1", color=(255, 0, 0, 255))
    )
    annotation2 = Annotation(
        words=[word3], label=LabelData(label="Label2", color=(0, 255, 0, 255))
    )
    sentence = Sentence(tokens=[annotation1, annotation2])

    assert sentence.get_right_neighbor(word1) == word2
    assert sentence.get_right_neighbor(word2) == word3
    assert sentence.get_right_neighbor(word3) is None


def test_get_word_parent():
    word1 = Word(word="word1")
    word2 = Word(word="word2")
    word3 = Word(word="word3")
    annotation1 = Annotation(
        words=[word1, word2], label=LabelData(label="Label1", color=(255, 0, 0, 255))
    )
    annotation2 = Annotation(
        words=[word3], label=LabelData(label="Label2", color=(0, 255, 0, 255))
    )
    sentence = Sentence(tokens=[annotation1, annotation2])

    assert sentence.get_word_parent(word1) == annotation1
    assert sentence.get_word_parent(word2) == annotation1
    assert sentence.get_word_parent(word3) == annotation2
    assert sentence.get_word_parent(Word(word="nonexistent")) is None


def test_to_list():
    word1 = Word(word="word1")
    word2 = Word(word="word2")
    word3 = Word(word="word3")
    annotation1 = Annotation(
        words=[word1, word2], label=LabelData(label="Label1", color=(255, 0, 0, 255))
    )
    annotation2 = Annotation(
        words=[word3], label=LabelData(label="Label2", color=(0, 255, 0, 255))
    )
    sentence = Sentence(tokens=[annotation1, annotation2])

    expected_list = ["word1", "word2", "word3", "B-Label1", "I-Label1", "B-Label2"]
    assert sentence.to_list() == expected_list
