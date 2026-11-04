"""Tests de la limpieza (función pura compartida por entrenamiento e inferencia)."""

import pytest

from fakenews.cleaning import build_document, clean_text, rule_hit_counts, rule_names

CASES = [
    "WASHINGTON (Reuters) - The U.S. Senate voted on Tuesday, officials told Reuters.",
    "Trump s speech was a disaster. Featured image via Getty Images",
    "Look at this pic.twitter.com/abc123 — John Doe (@jdoe) December 31, 2017 and more text here.",
    "Read the report at https://example.com/report?id=1 and www.site.org/page now",
    "BREAKING (VIDEO): WATCH: Senator says “no” to the bill’s new version",
    "Via: Breitbart News. Some content follows here.",
    "Tune in to the Alternate Current Radio Network (ACR) for another LIVE broadcast. Real content.",
    "This article was funded in part by SAP. It was independently created by the Reuters editorial staff.",
    "<p>Some &amp; HTML</p> text with​zero width",
    "Café, naïve, 日本語, emojis 😀🔥 stay.",
    "",
]


@pytest.mark.parametrize("text", CASES)
def test_idempotent(text):
    once = clean_text(text)
    assert clean_text(once) == once


def test_removes_agency_marks():
    out = clean_text(CASES[0])
    assert "Reuters" not in out and "WASHINGTON (" not in out
    assert out.startswith("The U.S. Senate voted")


def test_removes_photo_credits_twitter_urls():
    assert "Featured" not in clean_text(CASES[1]) and "Getty" not in clean_text(CASES[1])
    out = clean_text(CASES[2])
    assert "pic.twitter" not in out and "@jdoe" not in out and "2017" not in out
    out = clean_text(CASES[3])
    assert "http" not in out and "www" not in out and "example.com" not in out


def test_removes_media_tags_and_quotes():
    out = clean_text(CASES[4])
    assert "(VIDEO)" not in out and "WATCH:" not in out
    assert "“" not in out and "’" not in out and "'" not in out
    assert "BREAKING" in out  # el tono se conserva: solo se quitan marcas de plataforma


def test_apostrophe_normalization_aligns_sources():
    # Las reales usan ’ y en las falsas el apóstrofo quedó como espacio: ambas deben coincidir.
    assert clean_text("Trump’s plan") == clean_text("Trump s plan")


def test_removes_promos_and_disclaimers():
    assert "Alternate Current" not in clean_text(CASES[6])
    assert "Real content." in clean_text(CASES[6])
    assert "SAP" not in clean_text(CASES[7])
    assert "Via:" not in clean_text(CASES[5]) and "Some content follows here." in clean_text(CASES[5])


def test_unicode_preserved_and_html_removed():
    out = clean_text(CASES[9])
    assert "Café" in out and "naïve" in out and "日本語" in out and "😀" in out
    out = clean_text(CASES[8])
    assert "<p>" not in out and "&amp;" not in out and "​" not in out and " " not in out


def test_does_not_empty_normal_text():
    text = "The committee will meet next week to discuss the proposal in detail."
    assert clean_text(text) == text


@pytest.mark.parametrize("value", [None, 3, float("nan"), b"bytes"])
def test_non_string_inputs(value):
    assert clean_text(value) == ""


def test_minimal_mode_keeps_marks():
    out = clean_text(CASES[0], mode="minimal")
    assert "(Reuters)" in out


def test_invalid_mode():
    with pytest.raises(ValueError):
        clean_text("x", mode="otro")


def test_build_document_and_rule_reporting():
    doc = build_document("Title here", "WASHINGTON (Reuters) - Body text.")
    assert doc.split("\n")[0] == "Title here" and "Reuters" not in doc
    hits = rule_hit_counts(CASES[0])
    assert hits["agency_dateline"] == 1
    assert "site_promo" in rule_names()
