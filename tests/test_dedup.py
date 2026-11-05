"""Tests de duplicados exactos y casi-duplicados (MinHash + LSH)."""

from fakenews.dedup import (connected_groups, cross_set_near_duplicates, exact_key, jaccard,
                            near_duplicate_pairs, shingles)

BASE = ("the senate passed the infrastructure bill on tuesday after weeks of negotiations between "
        "the two parties and the white house said the president would sign it soon")


def test_exact_key_normalizes_case_and_punctuation():
    assert exact_key("Hello, World!") == exact_key("hello world")
    assert exact_key("a b") != exact_key("a c")


def test_shingles_deterministic():
    assert shingles(BASE) == shingles(BASE)
    assert jaccard(shingles(BASE), shingles(BASE)) == 1.0


def test_near_duplicates_found_and_distinct_texts_not():
    near = BASE + " today"
    other = "a completely different article about football results in europe and the weather " * 2
    pairs = near_duplicate_pairs([BASE, near, other], threshold=0.8)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    assert connected_groups(3, pairs) == [0, 0, 2]


def test_cross_set():
    pairs = cross_set_near_duplicates([BASE, "unrelated text " * 10], [BASE + " again"], threshold=0.8)
    assert [(i, j) for i, j, _ in pairs] == [(0, 0)]
