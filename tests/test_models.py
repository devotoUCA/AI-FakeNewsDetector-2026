"""Tests de modelos: determinismo, superar al trivial y pisos de F1."""

import numpy as np
import pytest

from fakenews import models as M
from fakenews.config import get_paths, load_config, read_json

from .conftest import needs_model


def _small_lr():
    pipe = M.logreg(1.0, (1, 1))
    pipe.set_params(tfidf__min_df=1, tfidf__max_df=1.0)
    return pipe


def test_training_is_deterministic(tiny_corpus):
    X, y = tiny_corpus
    a, b = _small_lr().fit(X, y), _small_lr().fit(X, y)
    assert np.array_equal(a.named_steps["clf"].coef_, b.named_steps["clf"].coef_)
    # El bosque se entrena con su configuración real (n_jobs=-1), que es determinista. Solo la predicción
    # va en un hilo: con varios, predict_proba suma los árboles en el orden en que terminan los hilos y
    # el resultado puede variar en el último bit (~1e-16). Así la comparación puede seguir siendo exacta.
    s1 = M.style_forest(5).fit(X, y).set_params(clf__n_jobs=1)
    s2 = M.style_forest(5).fit(X, y).set_params(clf__n_jobs=1)
    assert np.array_equal(M.p_fake(s1, X), M.p_fake(s2, X))
    # Control: otra semilla debe dar otro resultado, o el test no detectaría un random_state roto.
    s3 = M.style_forest(5).set_params(clf__random_state=43).fit(X, y).set_params(clf__n_jobs=1)
    assert not np.array_equal(M.p_fake(s1, X), M.p_fake(s3, X))


def test_dummy_predicts_prior(tiny_corpus):
    X, y = tiny_corpus
    d = M.dummy().fit(X, y)
    assert np.allclose(M.p_fake(d, X), np.mean(y))


def test_lr_beats_dummy_on_fixture(tiny_corpus):
    X, y = tiny_corpus
    acc = (_small_lr().fit(X, y).predict(X) == np.array(y)).mean()
    assert acc > max(np.mean(y), 1 - np.mean(y))


@needs_model
def test_every_model_beats_dummy_on_validation():
    r = read_json(get_paths().reports / "val_results.json")
    dummy_f1_fake = r["dummy"]["fake"]["f1"]
    dummy_acc = r["dummy"]["accuracy"]
    for row in r["tabla_val"]:
        if row["modelo"] == "dummy":
            continue
        assert row["f1_fake"] > dummy_f1_fake, row["modelo"]
        assert row["accuracy"] > dummy_acc, row["modelo"]


@needs_model
def test_productive_model_f1_floors_on_validation():
    floor = load_config()["evaluation"]["success_f1_per_class"]
    r = read_json(get_paths().reports / "val_results.json")
    prod = r["productivo"]["modelo"]
    f1_real, f1_fake = r[prod]["real"]["f1"], r[prod]["fake"]["f1"]
    if min(f1_real, f1_fake) <= floor:
        pytest.fail(f"F1 por clase por debajo del piso {floor}: real={f1_real:.3f}, falsa={f1_fake:.3f}")


@needs_model
def test_calibration_does_not_worsen_brier_on_validation():
    cal = read_json(get_paths().reports / "calibration_val.json")
    for name, info in cal.items():
        chosen = info[info["elegido"]]["brier"]
        reference = info["none"]["brier"] if "none" in info else min(
            v["brier"] for k, v in info.items() if isinstance(v, dict))
        assert chosen <= reference + 1e-12, name
