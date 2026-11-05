"""Contrato de `predict` con la compuerta de dominio, probabilidades extremas y retrocompatibilidad."""

import math

import pytest

from fakenews.predict import OUT_OF_DOMAIN, clip_public, display_probability
from fakenews.thresholds import LABEL_UNCERTAIN

from .conftest import needs_data
from .conftest_v3 import needs_v3_model

OLD_KEYS = {"label", "p_false", "confidence_level", "ambiguous", "top_terms", "warnings"}
NEW_KEYS = {"in_domain", "domain_score", "p_false_reference", "p_false_display", "cleaning_profile"}
FOREIGN = {
    "afirmacion_liar": "Says the unemployment rate doubled under the governor.",
    "espanol": "El gobierno anunció hoy un nuevo plan económico para las pequeñas empresas y un aumento del gasto "
               "en infraestructura durante el próximo año fiscal según informó el ministro.",
    "url": "https://example.com/noticia",
    "lista": "milk\neggs\nbread\nbutter\ncheese\napples",
    "vacio": "",
}


@pytest.fixture(scope="module")
def predictor():
    from fakenews.predict import Predictor

    return Predictor()


@pytest.mark.parametrize("p,txt", [(0.0, "< 1 %"), (0.004, "< 1 %"), (0.999, "> 99 %"), (1.0, "> 99 %"), (0.5, "50,0 %")])
def test_display_probability(p, txt):
    assert display_probability(p) == txt


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.2, float("nan")])
def test_public_probability_is_never_literal_0_or_1(p):
    out = clip_public(p)
    assert 0.0 < out < 1.0 and not math.isnan(out)


@needs_v3_model
@pytest.mark.parametrize("name", list(FOREIGN))
def test_foreign_texts_are_out_of_domain(predictor, name):
    out = predictor.predict(FOREIGN[name])
    assert OLD_KEYS | NEW_KEYS <= set(out)
    assert out["in_domain"] is False and out["label"] == LABEL_UNCERTAIN
    assert out["p_false"] is None and out["confidence_level"] == "baja"
    assert any(w.startswith(OUT_OF_DOMAIN) for w in out["warnings"]) or name == "vacio"


@needs_v3_model
@needs_data
def test_isot_validation_is_mostly_in_domain_and_outputs_are_clean(predictor):
    from fakenews.corpus import isot_raw, isot_split

    va = isot_split("val", "v1").sample(150, random_state=42)
    raw = isot_raw(va["id"])
    outs = [predictor.predict(f"{t}\n{b}") for t, b in zip(raw["title"], raw["text"])]
    inside = [o for o in outs if o["in_domain"]]
    assert len(inside) / len(outs) >= 0.9
    for o in inside:
        assert 0.0 < o["p_false"] < 1.0 and "disponible" not in o["p_false_display"]
        assert o["p_false"] == o["p_false_reference"]
    extreme = [o for o in inside if o["p_false"] <= 0.01 or o["p_false"] >= 0.99]
    assert extreme and all(o["p_false_display"] in ("< 1 %", "> 99 %") for o in extreme)


@needs_v3_model
def test_no_global_shrinkage_in_v3(predictor):
    """Corrección A1: dentro del dominio p_false es la calibrada (no hay contracción hacia la prior)."""
    assert predictor.ood_params is None and predictor.gate is not None


@needs_v3_model
def test_gate_is_deterministic(predictor):
    a, b = predictor.predict(FOREIGN["espanol"]), predictor.predict(FOREIGN["espanol"])
    assert a == b
