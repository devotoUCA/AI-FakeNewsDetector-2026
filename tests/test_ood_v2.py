"""OOD y abstención: textos de otro estilo no reciben confianza alta; la regla es determinista."""

import numpy as np
import pytest

from fakenews.ood import abstention_table, choose_tau, ood_score, shrink
from fakenews.thresholds import LABEL_UNCERTAIN

from .conftest_v2 import needs_v2_model

LIAR_LIKE = "Says the unemployment rate doubled under the governor."  # afirmación corta tipo LIAR (ilustrativa)


def test_shrink_and_score_are_deterministic():
    p = np.array([0.0, 0.5, 1.0])
    assert np.allclose(shrink(p, np.array([1.0, 1.0, 1.0]), 0.45), 0.45)
    assert np.allclose(shrink(p, np.zeros(3), 0.45), p)
    params = {"mean": [0, 0], "scale": [1, 1], "coef": [1.0, -1.0], "intercept": 0.0}
    f = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(ood_score(f, params), ood_score(f, params))
    assert ood_score(f, params)[0] > 0.5 > ood_score(f, params)[1]


def test_choose_tau_rule():
    s_in = np.linspace(0, 0.3, 100)
    s_ext = np.linspace(0.2, 1.0, 100)
    y_ext = np.r_[np.zeros(60), np.ones(40)].astype(int)
    p_ext = np.full(100, 0.9)  # siempre "falsa": peor que el trivial (real) afuera
    rows = abstention_table(s_in, s_ext, y_ext, p_ext, np.zeros(100, int), np.zeros(100), 0.5, 0)
    r = choose_tau(rows, 0.95)
    assert r["fila"]["cobertura_in"] >= 0.95


@pytest.fixture(scope="module")
def predictor():
    from fakenews.predict import Predictor

    # Estos tests cubren el comportamiento del artefacto de la iteración 2 (sigue reproducible).
    return Predictor(artifact="final_model_v2.joblib")


@needs_v2_model
@pytest.mark.parametrize("text", ["", "corto", "https://example.com/x", LIAR_LIKE,
                                  "El gobierno anunció hoy un nuevo plan económico para las pequeñas empresas "
                                  "y un aumento del gasto en infraestructura durante el próximo año fiscal."])
def test_other_style_is_uncertain_or_not_high(predictor, text):
    out = predictor.predict(text)
    assert out["label"] == LABEL_UNCERTAIN or out["confidence_level"] != "alta"


@needs_v2_model
def test_ood_flagged_never_high(predictor):
    if not predictor.ood_params:
        pytest.skip("modelo sin parámetros OOD")
    long_offtopic = ("The recipe calls for two cups of flour, a pinch of salt and butter. Knead the dough "
                     "for ten minutes, let it rest, then bake it at a moderate temperature until golden. ") * 4
    out = predictor.predict(long_offtopic)
    if out["ood_score"] is not None and out["ood_score"] >= predictor.ood_params["tau_cap"]:
        assert out["confidence_level"] != "alta"
    a, b = predictor.predict(long_offtopic), predictor.predict(long_offtopic)
    assert a == b  # determinista


@needs_v2_model
def test_predict_uses_artifact_profile(predictor):
    out = predictor.predict("The committee will meet next week to discuss the proposal in detail with "
                            "members of both parties and outside experts on the budget.")
    assert out["cleaning_profile"] == predictor.artifact["profile"]
    for key in ("label", "p_false", "confidence_level", "ambiguous", "top_terms", "warnings",
                "p_false_raw", "ood_score"):
        assert key in out
