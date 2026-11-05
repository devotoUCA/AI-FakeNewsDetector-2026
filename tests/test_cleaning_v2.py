"""Perfiles de limpieza v2/v3: idempotencia, contracciones, unicode, entradas raras."""

import pytest

from fakenews.cleaning import PROFILES, clean_text

SAMPLES = [
    "WASHINGTON (Reuters) - The U.S. Senate on Tuesday said it won’t vote, officials told Reuters.",
    "BREAKING!!! They don t want you to know!!! It s SHOCKING. https://x.com/a @user #MAGA",
    "He doesn't care. She can't. Don’t do it. 21st century, 1,200 people in May 2017.",
    "Café, naïve, 日本語, emojis 😀🔥 stay. ¿Qué tal?",
    "<p>Some &amp; HTML</p> text with​zero width",
    "", "   ", "!!! ??? ...", "@@@ ### $$$",
]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("text", SAMPLES)
def test_idempotent_all_profiles(profile, text):
    once = clean_text(text, profile=profile)
    assert clean_text(once, profile=profile) == once


@pytest.mark.parametrize("profile", ["v2", "v3"])
def test_contractions_collapse(profile):
    forms = ["They don't go.", "They don’t go.", "They don t go."]
    outs = {clean_text(f, profile=profile) for f in forms}
    assert len(outs) == 1 and "do not" in outs.pop()
    poss = {clean_text(f, profile=profile) for f in ["Trump’s plan", "Trump's plan", "Trump s plan"]}
    assert len(poss) == 1


def test_v2_neutral_tokens_and_dates():
    out = clean_text("Visit https://a.com/x or @bob #tag on Tuesday, Nov. 8, 2016 at 10 a.m. EDT.", profile="v2")
    assert "xurlx" in out and "xuserx" in out and "xhashtagx" in out
    assert "Tuesday" not in out and "Nov" not in out and "EDT" not in out and "2016" not in out
    assert "xnumx" in out and "xtimex" in out
    assert clean_text("The U.S. and U.K. agreed.", profile="v2") == "The US and UK agreed."


def test_v3_masks_agency_vocabulary_and_caps():
    out = clean_text("Officials said the SHOCKING plan failed!!! He told reporters. See https://a.com", profile="v3")
    low = out.lower()
    assert " said" not in low and "told" not in low and "reporters" not in low
    assert "shocking" in out and "SHOCKING" not in out and "!!!" not in out
    assert "xurlx" not in out


@pytest.mark.parametrize("profile", PROFILES)
def test_unicode_preserved_and_empty_inputs(profile):
    out = clean_text(SAMPLES[3], profile=profile)
    assert "Café" in out and "日本語" in out and "😀" in out
    for v in ("", "   ", None, 3):
        assert clean_text(v, profile=profile) == ""


def test_unknown_profile():
    with pytest.raises(ValueError):
        clean_text("x", profile="v9")


def test_v1_default_unchanged():
    # El perfil por defecto sigue siendo v1: los artefactos de la iteración 1 no cambian.
    assert clean_text(SAMPLES[0]) == clean_text(SAMPLES[0], profile="v1")
    assert "Tuesday" in clean_text(SAMPLES[0])
