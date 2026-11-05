"""Métricas de decisión y de probabilidad, intervalos bootstrap y test de McNemar."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, precision_recall_fscore_support)

EPS = 1e-15  # recorte de probabilidades para log-loss (evita log(0) en kNN/árboles)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           p_fake: np.ndarray | None = None) -> dict[str, Any]:
    """Accuracy, precisión/recall/F1 por clase, matriz de confusión y (si hay p) Brier/log-loss."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1],
                                                         zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "real": {"precision": float(prec[0]), "recall": float(rec[0]), "f1": float(f1[0]),
                 "support": int(sup[0])},
        "fake": {"precision": float(prec[1]), "recall": float(rec[1]), "f1": float(f1[1]),
                 "support": int(sup[1])},
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_rate": float(fp / max(1, fp + tn)),
    }
    if p_fake is not None:
        out.update(probability_metrics(y_true, p_fake))
    return out


def probability_metrics(y_true: np.ndarray, p_fake: np.ndarray, n_bins: int = 10
                        ) -> dict[str, float]:
    """Brier score, log-loss y ECE de la probabilidad de la clase falsa."""
    p = np.clip(np.asarray(p_fake, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y_true).astype(int)
    return {"brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "ece": expected_calibration_error(y, p, n_bins)}


def reliability_bins(y_true: np.ndarray, p_fake: np.ndarray, n_bins: int = 10
                     ) -> list[dict[str, float]]:
    """Tabla del diagrama de confiabilidad: por bin, probabilidad media vs. frecuencia real."""
    y, p = np.asarray(y_true).astype(int), np.asarray(p_fake, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({"bin_inf": float(edges[b]), "bin_sup": float(edges[b + 1]), "n": int(m.sum()),
                     "p_media": float(p[m].mean()) if m.any() else float("nan"),
                     "frac_falsas": float(y[m].mean()) if m.any() else float("nan")})
    return rows


def expected_calibration_error(y_true: np.ndarray, p_fake: np.ndarray, n_bins: int = 10) -> float:
    """ECE: promedio ponderado de |frecuencia real - probabilidad media| por bin."""
    rows = reliability_bins(y_true, p_fake, n_bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["frac_falsas"] - r["p_media"]) for r in rows if r["n"]) / max(1, n))


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, p_fake: np.ndarray | None,
                 n_resamples: int, seed: int, alpha: float = 0.05) -> dict[str, list[float]]:
    """Intervalos de confianza percentil (95 %) por bootstrap para las métricas principales."""
    rng = np.random.default_rng(seed)
    y, yp = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    p = None if p_fake is None else np.clip(np.asarray(p_fake, dtype=float), EPS, 1 - EPS)
    n = len(y)
    stats: dict[str, list[float]] = {k: [] for k in ("accuracy", "f1_fake", "f1_real", "brier")}
    for _ in range(n_resamples):
        i = rng.integers(0, n, n)
        yt, ypi = y[i], yp[i]
        stats["accuracy"].append(float((yt == ypi).mean()))
        stats["f1_fake"].append(_f1(yt, ypi, 1))
        stats["f1_real"].append(_f1(yt, ypi, 0))
        if p is not None:
            stats["brier"].append(float(np.mean((p[i] - yt) ** 2)))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))]
            for k, v in stats.items() if v}


def _f1(y: np.ndarray, yp: np.ndarray, pos: int) -> float:
    tp = np.sum((yp == pos) & (y == pos))
    fp = np.sum((yp == pos) & (y != pos))
    fn = np.sum((yp != pos) & (y == pos))
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    """Test de McNemar exacto (binomial) sobre los casos en que A y B difieren en acierto."""
    y = np.asarray(y_true).astype(int)
    a_ok, b_ok = np.asarray(pred_a) == y, np.asarray(pred_b) == y
    only_a, only_b = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    n = only_a + only_b
    p = float(binomtest(only_a, n, 0.5).pvalue) if n > 0 else 1.0
    return {"solo_A_acierta": only_a, "solo_B_acierta": only_b, "discordantes": n, "p_value": p}


def compact(m: dict[str, Any]) -> dict[str, float]:
    """Resumen plano de métricas (para tablas)."""
    out = {"accuracy": m["accuracy"], "precision_fake": m["fake"]["precision"],
           "recall_fake": m["fake"]["recall"], "f1_fake": m["fake"]["f1"],
           "precision_real": m["real"]["precision"], "recall_real": m["real"]["recall"],
           "f1_real": m["real"]["f1"]}
    for k in ("brier", "log_loss", "ece"):
        if k in m:
            out[k] = m[k]
    return out
