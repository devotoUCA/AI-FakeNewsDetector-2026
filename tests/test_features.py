"""Tests de representaciones: TF-IDF y rasgos de estilo."""

import numpy as np
import pytest

from fakenews.features import (STYLE_FEATURES, StyleFeatures, english_stopword_ratio, known_vocab_ratio,
                               make_tfidf, non_ascii_letter_ratio, style_vector)


def test_style_known_values():
    v = dict(zip(STYLE_FEATURES, style_vector("WOW this is SHOCKING! Really?")))
    # palabras: WOW this is SHOCKING Really -> 5
    assert v["log_n_words"] == pytest.approx(np.log1p(5))
    assert v["allcaps_word_ratio"] == pytest.approx(2 / 5)
    assert v["exclam_per_100w"] == pytest.approx(100 / 5)
    assert v["question_per_100w"] == pytest.approx(100 / 5)
    assert v["sensational_ratio"] == pytest.approx(2 / 5)  # wow, shocking


def test_style_attribution_and_empty():
    v = dict(zip(STYLE_FEATURES, style_vector("He said the officials told reporters")))
    assert v["attribution_ratio"] == pytest.approx(3 / 6)  # said, officials, told
    assert np.all(style_vector("") == 0)


def test_style_transformer_shape_and_determinism(tiny_corpus):
    X, _ = tiny_corpus
    a, b = StyleFeatures().transform(X), StyleFeatures().transform(X)
    assert a.shape == (len(X), len(STYLE_FEATURES))
    assert np.array_equal(a, b)


def test_tfidf_shape_and_determinism(tiny_corpus):
    X, _ = tiny_corpus
    v1 = make_tfidf((1, 2)).set_params(min_df=1, max_df=1.0)
    v2 = make_tfidf((1, 2)).set_params(min_df=1, max_df=1.0)
    A, B = v1.fit_transform(X), v2.fit_transform(X)
    assert A.shape[0] == len(X) and (A != B).nnz == 0
    norms = np.sqrt(A.multiply(A).sum(axis=1)).A1
    assert np.allclose(norms, 1.0, atol=1e-5)  # vectores L2-normalizados (distancia coseno)


def test_language_heuristics():
    en = "The government said that it will review the decision with the members of the committee."
    es = "El gobierno dijo que revisará la decisión con los miembros de la comisión de manera urgente."
    assert english_stopword_ratio(en) > english_stopword_ratio(es)
    assert non_ascii_letter_ratio("Привет мир") == 1.0
    assert known_vocab_ratio("the cat", {"the"}) == 0.5
