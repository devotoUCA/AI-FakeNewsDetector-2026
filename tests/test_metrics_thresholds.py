"""Tests de métricas, umbral, zona ambigua y nivel de confianza."""

import numpy as np
import pytest

from fakenews.metrics import (bootstrap_ci, classification_metrics, expected_calibration_error,
                              mcnemar_exact, probability_metrics)
from fakenews.thresholds import (LABEL_FAKE, LABEL_REAL, LABEL_UNCERTAIN, choose_confidence_cuts,
                                 choose_threshold, confidence_level, coverage_curve, decide)


def test_classification_metrics_known_case():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, np.array([0, 1, 1, 1]), np.array([0.1, 0.6, 0.9, 0.8]))
    assert m["accuracy"] == 0.75
    assert m["fake"]["recall"] == 1.0 and m["fake"]["precision"] == pytest.approx(2 / 3)
    assert m["confusion"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert m["brier"] == pytest.approx(np.mean([0.01, 0.36, 0.01, 0.04]))


def test_probability_metrics_bounds_and_clip():
    m = probability_metrics(np.array([0, 1]), np.array([0.0, 1.0]))
    assert m["brier"] == pytest.approx(0.0, abs=1e-12) and np.isfinite(m["log_loss"])
    assert expected_calibration_error(np.array([0, 1] * 50), np.full(100, 0.5)) == pytest.approx(0.0)


def test_bootstrap_and_mcnemar():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    ci = bootstrap_ci(y, y, y.astype(float), 200, 42)
    assert ci["accuracy"] == [1.0, 1.0]
    r = mcnemar_exact(y, y, 1 - y)
    assert r["solo_A_acierta"] == 300 and r["p_value"] < 1e-10


@pytest.mark.parametrize("p,label", [(0.05, LABEL_REAL), (0.39, LABEL_REAL), (0.4, LABEL_UNCERTAIN),
                                     (0.5, LABEL_UNCERTAIN), (0.6, LABEL_UNCERTAIN), (0.61, LABEL_FAKE),
                                     (0.99, LABEL_FAKE)])
def test_decide_coherent_with_threshold_and_zone(p, label):
    assert decide(p, 0.5, 0.4, 0.6) == label


def test_threshold_choice_respects_fpr_and_prefers_05_on_ties():
    y = np.array([0] * 50 + [1] * 50)
    p = np.r_[np.linspace(0.0, 0.3, 50), np.linspace(0.7, 1.0, 50)]
    r = choose_threshold(y, p, [0.2, 0.8, 0.01], max_fpr=0.1)
    assert r["umbral"] == pytest.approx(0.5)  # separación perfecta: empate amplio -> 0,5
    assert r["elegido"]["fpr"] <= 0.1


def test_confidence_levels():
    cuts = {"baja_si_conf_menor_a": 0.6, "alta_si_conf_mayor_o_igual_a": 0.9}
    assert confidence_level(0.5, cuts) == "baja"
    assert confidence_level(0.25, cuts) == "media"
    assert confidence_level(0.95, cuts) == "alta" and confidence_level(0.03, cuts) == "alta"


def test_confidence_cuts_use_local_accuracy():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)  # probabilidades perfectamente calibradas
    r = choose_confidence_cuts(y, p, 0.5, 0.4, 0.6, [0.6, 0.99, 0.01], target_acc=0.9)
    # Calibrado => accuracy local ~ confianza: el corte "alta" debe quedar cerca de 0,9.
    assert 0.85 <= r["cortes"]["alta_si_conf_mayor_o_igual_a"] <= 0.95


def test_coverage_curve_monotone_coverage():
    y = np.array([0, 1] * 100)
    p = np.linspace(0, 1, 200)
    rows = coverage_curve(y, p, 0.5)
    cov = [r["cobertura"] for r in rows]
    assert cov[0] == 1.0 and all(a >= b for a, b in zip(cov, cov[1:]))
