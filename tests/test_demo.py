"""La lógica de la demo se prueba sin levantar el servidor de Streamlit."""

from fakenews.demo_logic import PERMANENT_NOTICE, analyze, example_texts, load_predictor

from .conftest import needs_data, needs_model
from .test_predict import ARTICLE


def test_notice_mentions_limits():
    for w in ("no un veredicto", "Solo inglés", "sátira"):
        assert w in PERMANENT_NOTICE


def test_missing_model_gives_actionable_message(tmp_path, monkeypatch):
    from fakenews import demo_logic
    from fakenews.predict import Predictor

    monkeypatch.setattr(demo_logic, "Predictor", lambda: Predictor(models_dir=tmp_path))
    pred, err = load_predictor()
    assert pred is None and "make train" in err


@needs_model
def test_analyze_returns_label_and_probability():
    pred, err = load_predictor()
    assert err is None
    r = analyze(pred, ARTICLE)
    assert r["titulo_resultado"] and r["p_false"] is not None
    assert 0 <= r["barra"] <= 1 and r["confianza"] in ("Alta", "Media", "Baja")
    empty = analyze(pred, "")
    assert empty["p_false"] is None and empty["zona_ambigua"] and empty["advertencias"]


@needs_data
def test_examples_come_from_test_set_and_are_marked():
    ex = example_texts()
    assert len(ex) == 4 and all("dataset de test" in e["nombre"] for e in ex)
