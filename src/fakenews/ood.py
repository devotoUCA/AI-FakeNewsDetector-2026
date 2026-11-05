"""Puntaje de fuera-de-distribución (OOD) y abstención (iteración 2, punto E).

Rasgos baratos, sin dependencias pesadas, calculados con el índice de recuperación del modelo
(TF-IDF de unigramas sobre ISOT train):
* similitud coseno máxima con las noticias de entrenamiento y media de los 5 vecinos,
* cobertura de vocabulario (fracción de tokens conocidos),
* largo (log de palabras).

Un detector logístico de 4 rasgos se ajusta para separar ISOT-val (en distribución) de los
conjuntos externos de DESARROLLO (WELFake-dev + LIAR-valid). Su salida es el puntaje OOD en
[0, 1]. Con él: (a) la probabilidad se contrae hacia la prior: p' = prior + (p - prior)(1 - s);
(b) si s >= tau el sistema se abstiene ("incierta"); (c) si s >= tau_cap la confianza no puede
ser "alta". tau se elige en desarrollo; el holdout final nunca participa.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .cleaning import word_count
from .features import known_vocab_ratio

OOD_FEATURES = ["sim_max", "sim_media_5", "vocab_conocido", "log_palabras"]


def ood_features(texts: list[str], retrieval: dict[str, Any], vocabulary: Any) -> np.ndarray:
    """Matriz de rasgos OOD (n x 4) para textos ya limpios con el perfil del modelo."""
    pipe = retrieval["pipeline"]
    X = pipe.named_steps["tfidf"].transform(texts)
    out = np.zeros((len(texts), len(OOD_FEATURES)))
    nonempty = np.asarray(X.getnnz(axis=1) > 0)
    if nonempty.any():
        dist, _ = pipe.named_steps["clf"].kneighbors(X[nonempty], n_neighbors=5)
        sims = 1 - dist
        out[nonempty, 0] = sims[:, 0]
        out[nonempty, 1] = sims.mean(axis=1)
    out[:, 2] = [known_vocab_ratio(t, vocabulary) for t in texts]
    out[:, 3] = [np.log1p(word_count(t)) for t in texts]
    return out


def fit_ood_detector(f_in: np.ndarray, f_out: np.ndarray, seed: int) -> dict[str, Any]:
    """Ajusta el detector (0 = en distribución, 1 = externo). Devuelve parámetros serializables."""
    X = np.vstack([f_in, f_out])
    y = np.r_[np.zeros(len(f_in)), np.ones(len(f_out))]
    sc = StandardScaler().fit(X)
    # Pesos balanceados: los dos grupos cuentan lo mismo aunque tengan distinto tamaño.
    clf = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000).fit(sc.transform(X), y)
    return {"features": OOD_FEATURES, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "coef": clf.coef_.ravel().tolist(), "intercept": float(clf.intercept_[0])}


def ood_score(feats: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Puntaje OOD en [0, 1] (determinista, sin sklearn en inferencia)."""
    z = (feats - np.asarray(params["mean"])) / np.asarray(params["scale"])
    logit = z @ np.asarray(params["coef"]) + params["intercept"]
    return 1.0 / (1.0 + np.exp(-logit))


def shrink(p: np.ndarray, s: np.ndarray, prior: float) -> np.ndarray:
    """Contrae la probabilidad hacia la prior en proporción al puntaje OOD."""
    return prior + (np.asarray(p) - prior) * (1.0 - np.asarray(s))


def abstention_table(s_in: np.ndarray, s_ext: np.ndarray, y_ext: np.ndarray, p_ext: np.ndarray,
                     y_in: np.ndarray, p_in: np.ndarray, threshold: float, majority: int,
                     grid: np.ndarray | None = None) -> list[dict[str, float]]:
    """Para cada tau: cobertura in-domain y externa, accuracy de lo respondido y del trivial."""
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2) if grid is None else grid
    rows = []
    for tau in grid:
        ans_in, ans_ext = s_in < tau, s_ext < tau
        pred_ext = (p_ext >= threshold).astype(int)
        pred_in = (p_in >= threshold).astype(int)
        rows.append({
            "tau": float(tau), "cobertura_in": float(ans_in.mean()), "cobertura_ext": float(ans_ext.mean()),
            "n_ext_respondidos": int(ans_ext.sum()),
            "accuracy_in_respondidos": float((pred_in[ans_in] == y_in[ans_in]).mean()) if ans_in.any() else float("nan"),
            "accuracy_ext_respondidos": float((pred_ext[ans_ext] == y_ext[ans_ext]).mean()) if ans_ext.any() else float("nan"),
            "trivial_ext_respondidos": float((y_ext[ans_ext] == majority).mean()) if ans_ext.any() else float("nan"),
        })
    return rows


def choose_tau(rows: list[dict[str, float]], min_in_coverage: float, min_answered: int = 30) -> dict[str, Any]:
    """Regla: el tau MÁS PERMISIVO que (a) responde >= `min_in_coverage` de ISOT-val y (b) en el
    externo de desarrollo, sobre lo respondido, no es peor que el trivial. Si ninguno cumple (b),
    el tau más estricto que cumple (a) (máxima abstención compatible con seguir siendo útil)."""
    ok_a = [r for r in rows if r["cobertura_in"] >= min_in_coverage]
    if not ok_a:
        ok_a = [max(rows, key=lambda r: r["cobertura_in"])]

    def ok_b(r: dict[str, float]) -> bool:
        if r["n_ext_respondidos"] < min_answered:
            return True  # casi no responde afuera: no puede ser peor que el trivial
        return r["accuracy_ext_respondidos"] >= r["trivial_ext_respondidos"]

    both = [r for r in ok_a if ok_b(r)]
    if both:
        chosen = max(both, key=lambda r: r["tau"])
        rule = "más permisivo que cumple cobertura in-domain y no-peor-que-trivial externo"
    else:
        chosen = min(ok_a, key=lambda r: r["tau"])
        rule = "ningún tau cumple no-peor-que-trivial: se toma el más estricto con cobertura in-domain"
    return {"tau": chosen["tau"], "fila": chosen, "regla": rule}
