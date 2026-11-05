"""Perfil v2_1: espacios tras puntuación, idempotencia, unicode y casos borde; v2 no cambia."""

import pytest

from fakenews.cleaning import clean_text
from fakenews.format_shortcuts import auc_columns, to_format_classes

CASES = ["He left.The vote failed .It was , odd ( yes ) ... really…  Trump said:No.  (  )  — ok – fine",
         "¿Qué? Café 😀 日本語.", "", "   ", ". , .", "!!!", "J.K. Rowling, e.g. this.", "a.b.c.d", "https://a.com/x.y",
         "line one.\nLine two.Line three", "...", "()", "[ ]", "Trump’s plan.The U.S. said it won’t.",
         " hard space.Next"]


@pytest.mark.parametrize("text", CASES)
def test_idempotent(text):
    once = clean_text(text, profile="v2_1")
    assert clean_text(once, profile="v2_1") == once


def test_sentence_spacing_normalized():
    assert clean_text("He left.The vote failed", profile="v2_1") == clean_text("He left. The vote failed", profile="v2_1")
    assert clean_text("Yes,no.Why?Because;ok", profile="v2_1") == "Yes, no. Why? Because; ok"


def test_spaces_before_punctuation_and_empties():
    out = clean_text("It was , odd . ( yes ) ( ) ok", profile="v2_1")
    assert " ," not in out and " ." not in out and "( " not in out and " )" not in out and "()" not in out


def test_dashes_and_ellipsis():
    assert clean_text("Wait… what — no – yes...", profile="v2_1").count("…") == 0
    out = clean_text("one — two – three", profile="v2_1")
    assert "—" not in out and "–" not in out


def test_unicode_preserved():
    out = clean_text(CASES[1], profile="v2_1")
    assert "Café" in out and "日本語" in out and "😀" in out


def test_v2_is_not_modified():
    # `v2` debe seguir reproduciendo los resultados ya reportados: conserva el pegado.
    assert ".The" in clean_text("He left.The vote", profile="v2")


def test_format_classes_and_vectorized_auc():
    import numpy as np

    assert to_format_classes("Ab1 .") == "Aa0_."
    X = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]])
    y = np.array([0, 0, 1, 1])
    assert list(auc_columns(X, y)) == [1.0, 0.5]
