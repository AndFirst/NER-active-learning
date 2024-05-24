from app.data_types import Word


def test_hash():
    word1 = Word(word="test")
    word2 = Word(word="test")
    word3 = Word(word="different")

    assert hash(word1) == hash(word1)
    assert hash(word1) != hash(word2)
    assert hash(word1) != hash(word3)


def test_equality():
    word1 = Word(word="test")
    word2 = Word(word="test")
    word3 = Word(word="different")

    assert word1 == word1
    assert word1 != word2
    assert word1 != word3


def test_ordering():
    word1 = Word(word="apple")
    word2 = Word(word="banana")

    assert word1 < word2
    assert word2 > word1
