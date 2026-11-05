"""Explicación de las predicciones: términos que más empujan hacia "falsa" o hacia "real".

Para un modelo lineal sobre TF-IDF, la contribución de un término presente en el texto es
`peso * valor_tfidf` (acta, formulario 11: peso positivo empuja a "falsa", negativo a "real").
Si el modelo está calibrado por validación cruzada (5 modelos), se promedian los 5.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from .features import STYLE_FEATURES, StyleFeatures


def _weights(clf: Any) -> np.ndarray | None:
    if hasattr(clf, "coef_"):
        return np.asarray(clf.coef_).ravel()
    if hasattr(clf, "feature_log_prob_"):  # Naive Bayes: log-ratio de probabilidades por término
        flp = clf.feature_log_prob_
        return np.asarray(flp[1] - flp[0]).ravel()
    return None


def linear_parts(model: Any) -> list[tuple[TfidfVectorizer, np.ndarray]]:
    """Extrae los pares (vectorizador, pesos) de cualquier combinación de pipelines lineales."""
    if isinstance(model, CalibratedClassifierCV):
        parts = []
        for cc in model.calibrated_classifiers_:
            parts.extend(linear_parts(cc.estimator))
        return parts
    if isinstance(model, StackingClassifier):
        return linear_parts(model.named_estimators_["text"])
    if isinstance(model, Pipeline):
        vec = next((s for _, s in model.steps if isinstance(s, TfidfVectorizer)), None)
        w = _weights(model.steps[-1][1])
        if vec is not None and w is not None and len(w) == len(vec.vocabulary_):
            return [(vec, w)]
    return []


def style_part(model: Any) -> bool:
    """True si el modelo incluye un componente de rasgos de estilo."""
    if isinstance(model, StackingClassifier):
        return True
    if isinstance(model, CalibratedClassifierCV):
        return style_part(model.calibrated_classifiers_[0].estimator)
    if isinstance(model, Pipeline):
        return any(isinstance(s, StyleFeatures) for _, s in model.steps)
    return False


def term_contributions(model: Any, text: str, top_k: int = 8) -> dict[str, list[dict[str, float]]]:
    """Términos presentes en el texto con mayor contribución hacia cada clase."""
    parts = linear_parts(model)
    if not parts:
        return {"hacia_falsa": [], "hacia_real": []}
    acc: dict[str, float] = defaultdict(float)
    for vec, w in parts:
        x = vec.transform([text])
        names = vec.get_feature_names_out()
        for j, v in zip(x.indices, x.data):
            acc[names[j]] += float(v * w[j]) / len(parts)
    items = sorted(acc.items(), key=lambda kv: kv[1])
    fake = [{"termino": t, "peso": round(c, 4)} for t, c in reversed(items) if c > 0][:top_k]
    real = [{"termino": t, "peso": round(c, 4)} for t, c in items if c < 0][:top_k]
    return {"hacia_falsa": fake, "hacia_real": real}


def global_top_terms(model: Any, top_k: int = 30) -> dict[str, list[dict[str, float]]]:
    """Términos con mayor peso global del modelo (para detectar atajos, acta formulario 3)."""
    parts = linear_parts(model)
    acc: dict[str, float] = defaultdict(float)
    for vec, w in parts:
        for name, j in vec.vocabulary_.items():
            acc[name] += float(w[j]) / len(parts)
    items = sorted(acc.items(), key=lambda kv: kv[1])
    return {"hacia_falsa": [{"termino": t, "peso": round(c, 4)} for t, c in reversed(items[-top_k:])],
            "hacia_real": [{"termino": t, "peso": round(c, 4)} for t, c in items[:top_k]]}


def style_summary(text: str) -> dict[str, float]:
    """Valores de los rasgos de estilo del texto (para mostrar junto a la explicación)."""
    vals = StyleFeatures().transform([text])[0]
    return {k: round(float(v), 4) for k, v in zip(STYLE_FEATURES, vals)}
