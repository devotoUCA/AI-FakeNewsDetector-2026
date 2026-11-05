"""Atajos: ningún rasgo de atajo conocido entra al vector de entrada del modelo final."""

import joblib

from fakenews.cleaning import clean_text
from fakenews.config import get_paths
from fakenews.evaluate import SOURCE_TOKENS
from fakenews.explain import global_top_terms, linear_parts

from .conftest_v2 import needs_v2_model


@needs_v2_model
def test_metadata_and_marks_never_enter_input():
    art = joblib.load(get_paths().models / "final_model_v2.joblib")
    prof = art["profile"]
    text = ("WASHINGTON (Reuters) - On Tuesday, Nov. 8, the senator said “no”. Featured image via Getty "
            "Images. pic.twitter.com/x subject: politicsNews date: December 31, 2017 ")
    out = clean_text(text, profile=prof)
    for bad in ("(Reuters)", "Reuters", "“", "”", "Getty", "pic.twitter"):
        assert bad not in out, bad
    if prof in ("v2", "v3"):
        assert "Tuesday" not in out and "2017" not in out
    # El modelo recibe solo texto: un vectorizador TF-IDF sobre una lista de strings.
    parts = linear_parts(art["model"])
    assert parts, "el modelo final debe ser lineal sobre TF-IDF para poder auditarse"
    vocab = set(parts[0][0].vocabulary_)
    for meta in ("politicsnews", "worldnews", "left-news"):
        assert meta not in vocab


@needs_v2_model
def test_top_terms_exclude_known_source_marks():
    art = joblib.load(get_paths().models / "final_model_v2.joblib")
    top = global_top_terms(art["model"], 50)
    terms = {d["termino"] for d in top["hacia_falsa"] + top["hacia_real"]}
    assert not (terms & SOURCE_TOKENS), sorted(terms & SOURCE_TOKENS)
    if art["profile"] in ("v2", "v3"):
        weekdays = {"monday", "tuesday", "wednesday", "thursday", "friday", "on tuesday", "on wednesday"}
        assert not (terms & weekdays)
