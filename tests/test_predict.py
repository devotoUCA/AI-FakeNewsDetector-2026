"""Contrato de `predict` y casos borde: nunca debe lanzar una excepción sin controlar."""

import pytest

from fakenews.thresholds import CONFIDENCE_LEVELS, LABEL_FAKE, LABEL_REAL, LABEL_UNCERTAIN

from .conftest import needs_model

ARTICLE = (
    "The Senate on Tuesday approved a bill to fund the government through December, officials said, "
    "averting a shutdown that would have affected hundreds of thousands of federal workers. The "
    "measure now goes to the House of Representatives, where leaders said they expected a vote later "
    "this week. Lawmakers from both parties said the agreement was a compromise that left many issues "
    "unresolved, including disaster aid and the debt ceiling, which will be debated next month."
)
EDGE_CASES = {
    "vacio": "",
    "espacios": "   \n\t  ",
    "un_caracter": "a",
    "solo_url": "https://example.com/some/article?id=12",
    "emojis_unicode": "😀🔥 日本語のテキスト Привет мир ✨",
    "espanol": ("El gobierno anunció hoy un nuevo plan económico que incluye una reducción de impuestos "
                "para las pequeñas empresas y un aumento del gasto en infraestructura para el próximo año."),
    "larguisimo": ARTICLE * 120,
    "html_markdown": "<div><h1>Title</h1><p>**Bold** text with [link](http://x.com)</p></div> " + ARTICLE,
    "saltos_raros": ARTICLE.replace(" ", "\r\n   ", 30),
    "solo_numeros": "123 456 789 " * 10,
}


@pytest.fixture(scope="module")
def predictor():
    from fakenews.predict import Predictor

    return Predictor()


def _check_contract(out):
    for key in ("label", "p_false", "confidence_level", "ambiguous", "top_terms", "warnings"):
        assert key in out
    assert out["label"] in (LABEL_REAL, LABEL_FAKE, LABEL_UNCERTAIN)
    assert out["confidence_level"] in CONFIDENCE_LEVELS
    assert isinstance(out["ambiguous"], bool)
    assert isinstance(out["warnings"], list) and all(isinstance(w, str) for w in out["warnings"])
    assert set(out["top_terms"]) == {"hacia_falsa", "hacia_real"}
    if out["p_false"] is not None:
        assert isinstance(out["p_false"], float) and 0.0 <= out["p_false"] <= 1.0


@needs_model
def test_contract_on_article(predictor):
    out = predictor.predict(ARTICLE, with_similar=True)
    _check_contract(out)
    assert out["p_false"] is not None and out["warnings"] == []
    assert len(out["similar_examples"]) == 3
    # coherencia etiqueta / umbral / zona
    p = out["p_false"]
    if predictor.low <= p <= predictor.high:
        assert out["label"] == LABEL_UNCERTAIN and out["ambiguous"]
    else:
        assert out["label"] == (LABEL_FAKE if p >= predictor.threshold else LABEL_REAL)


@needs_model
@pytest.mark.parametrize("name", list(EDGE_CASES))
def test_edge_cases_never_raise(predictor, name):
    out = predictor.predict(EDGE_CASES[name])
    _check_contract(out)


@needs_model
@pytest.mark.parametrize("name", ["vacio", "espacios", "un_caracter", "solo_url", "emojis_unicode",
                                  "espanol", "solo_numeros"])
def test_out_of_distribution_is_uncertain_with_warning(predictor, name):
    out = predictor.predict(EDGE_CASES[name])
    assert out["label"] == LABEL_UNCERTAIN
    assert out["confidence_level"] == "baja"
    assert out["warnings"], name


@needs_model
def test_long_text_warns_and_lowers_confidence(predictor):
    out = predictor.predict(EDGE_CASES["larguisimo"])
    assert any("largo" in w for w in out["warnings"])
    assert out["confidence_level"] in ("media", "baja")


@needs_model
@pytest.mark.parametrize("value", [None, 42, ["lista"], {"a": 1}])
def test_non_string_input(predictor, value):
    out = predictor.predict(value)
    _check_contract(out)
    assert out["label"] == LABEL_UNCERTAIN


@needs_model
def test_module_level_predict():
    from fakenews.predict import predict

    _check_contract(predict(ARTICLE))
