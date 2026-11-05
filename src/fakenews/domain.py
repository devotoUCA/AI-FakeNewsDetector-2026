"""Compuerta de dominio de validez (iteración 3, punto B).

Razonamiento: el modelo no tiene habilidad fuera de ISOT, así que ninguna regla basada en "qué tan
seguro está" puede superar al trivial afuera. Lo honesto es decir cuándo un texto NO se parece a los
datos en los que el modelo es válido y no emitir veredicto. La compuerta mide pertenencia al dominio,
no veracidad: se evalúa por AUC de pertenencia y por detección de fuera-de-dominio a una retención
fija de ISOT val.

Detectores comparados:
(a) `vecindad`: logística sobre similitud con el entrenamiento, vocabulario y largo.
(b) `clasificador`: TF-IDF + regresión logística (positivos ISOT-train; negativos WELFake dev-train
    + LIAR-train).
(c) `combinado`: logística sobre (a) y el logit de (b).

Particiones (semilla fija): WELFake dev -> train 60 % / combinación 20 % / evaluación 20 %;
LIAR valid e ISOT val -> combinación 50 % / evaluación 50 %. Ningún holdout participa.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import get_paths, load_config, write_json
from .corpus import isot_split, liar_split, welfake_split
from .ood import ood_features

LOGGER = logging.getLogger(__name__)
GATE_FILE = "domain_gate_v3.json.gz"


def _gate_vectorizer(**kw: Any) -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1, 2), strip_accents="unicode",
                           dtype=np.float32, **kw)


def fit_text_classifier(pos: list[str], neg: list[str], seed: int, max_features: int) -> dict[str, Any]:
    """Clasificador de dominio (positivos = en dominio). Devuelve parámetros serializables."""
    vec = _gate_vectorizer(min_df=5, max_df=0.9, max_features=max_features)
    X = vec.fit_transform(pos + neg)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed).fit(X, y)
    names = vec.get_feature_names_out().tolist()
    return {"terms": names, "idf": vec.idf_.astype(float).round(5).tolist(),
            "coef": clf.coef_.ravel().astype(float).round(5).tolist(), "intercept": float(clf.intercept_[0])}


def text_logit(texts: list[str], params: dict[str, Any]) -> np.ndarray:
    """Logit de pertenencia del clasificador de dominio (reconstruido desde los parámetros)."""
    vec = _gate_vectorizer(vocabulary={t: i for i, t in enumerate(params["terms"])})
    vec.idf_ = np.asarray(params["idf"], dtype=np.float32)
    X = vec.transform(texts)
    return np.asarray(X @ np.asarray(params["coef"], dtype=np.float32)).ravel() + params["intercept"]


def _fit_logistic(F: np.ndarray, y: np.ndarray, seed: int) -> dict[str, Any]:
    sc = StandardScaler().fit(F)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed).fit(sc.transform(F), y)
    return {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": clf.coef_.ravel().tolist(),
            "intercept": float(clf.intercept_[0])}


def _logit(F: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    return ((F - np.asarray(p["mean"])) / np.asarray(p["scale"])) @ np.asarray(p["coef"]) + p["intercept"]


def domain_score(texts: list[str], gate: dict[str, Any], retrieval: dict[str, Any] | None, vocabulary: Any) -> np.ndarray:
    """Puntaje de pertenencia al dominio en (0, 1): más alto = más parecido a ISOT."""
    kind = gate["detector"]
    if kind == "clasificador":
        z = text_logit(texts, gate["clasificador"])
    else:
        F = ood_features(texts, retrieval, vocabulary)
        if kind == "vecindad":
            z = _logit(F, gate["vecindad"])
        else:
            z = _logit(np.c_[F, text_logit(texts, gate["clasificador"])], gate["combinado"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def detection_at_retention(s_in: np.ndarray, s_out: np.ndarray, retention: float) -> dict[str, float]:
    """Umbral que retiene `retention` de los textos en dominio y fracción de externos detectada."""
    thr = float(np.quantile(s_in, 1 - retention))
    return {"retencion": retention, "umbral": thr, "retencion_real": float((s_in >= thr).mean()),
            "deteccion_fuera_de_dominio": float((s_out < thr).mean())}


def _auc(s_in: np.ndarray, s_out: np.ndarray) -> float:
    return float(roc_auc_score(np.r_[np.ones(len(s_in)), np.zeros(len(s_out))], np.r_[s_in, s_out]))


def _splits(profile: str, seed: int) -> dict[str, list[str]]:
    wel = welfake_split("dev", profile)
    w_train, w_rest = train_test_split(wel["doc"].tolist(), train_size=0.6, random_state=seed)
    w_comb, w_eval = train_test_split(w_rest, train_size=0.5, random_state=seed)
    lv = liar_split("valid", profile)["doc"].tolist()
    l_comb, l_eval = train_test_split(lv, train_size=0.5, random_state=seed)
    va = isot_split("val", profile)["doc"].tolist()
    i_comb, i_eval = train_test_split(va, train_size=0.5, random_state=seed)
    return {"w_train": w_train, "w_comb": w_comb, "w_eval": w_eval, "l_train": liar_split("train", profile)["doc"].tolist(),
            "l_comb": l_comb, "l_eval": l_eval, "i_train": isot_split("train", profile)["doc"].tolist(),
            "i_comb": i_comb, "i_eval": i_eval}


def fit_domain_gate(profile: str, retrieval: dict[str, Any], vocabulary: Any) -> dict[str, Any]:
    """Compara los tres detectores, elige uno y fija el umbral. Devuelve la compuerta y el reporte."""
    cfg, paths = load_config(), get_paths()
    seed, dg = cfg["seed"], cfg["domain_gate"]
    d = _splits(profile, seed)
    clf = fit_text_classifier(d["i_train"], d["w_train"] + d["l_train"], seed, dg["max_features"])

    def feats(texts: list[str]) -> np.ndarray:
        return ood_features(texts, retrieval, vocabulary)

    F = {k: feats(d[k]) for k in ("i_comb", "w_comb", "l_comb", "i_eval", "w_eval", "l_eval")}
    L = {k: text_logit(d[k], clf) for k in F}
    y_comb = np.r_[np.ones(len(d["i_comb"])), np.zeros(len(d["w_comb"]) + len(d["l_comb"]))]
    vec_p = _fit_logistic(np.vstack([F["i_comb"], F["w_comb"], F["l_comb"]]), y_comb, seed)
    comb_p = _fit_logistic(np.c_[np.vstack([F["i_comb"], F["w_comb"], F["l_comb"]]),
                                 np.r_[L["i_comb"], L["w_comb"], L["l_comb"]]], y_comb, seed)
    scores = {
        "vecindad": {k: _logit(F[k], vec_p) for k in ("i_eval", "w_eval", "l_eval")},
        "clasificador": {k: L[k] for k in ("i_eval", "w_eval", "l_eval")},
        "combinado": {k: _logit(np.c_[F[k], L[k]], comb_p) for k in ("i_eval", "w_eval", "l_eval")},
    }
    rep: dict[str, Any] = {"detectores": {}}
    for name, s in scores.items():
        s_in, s_w, s_l = s["i_eval"], s["w_eval"], s["l_eval"]
        s_out = np.r_[s_w, s_l]
        rep["detectores"][name] = {
            "auc_pertenencia": _auc(s_in, s_out), "auc_vs_welfake": _auc(s_in, s_w), "auc_vs_liar": _auc(s_in, s_l),
            "por_retencion": [detection_at_retention(s_in, s_out, r) | {
                "deteccion_welfake": float((s_w < np.quantile(s_in, 1 - r)).mean()),
                "deteccion_liar": float((s_l < np.quantile(s_in, 1 - r)).mean())} for r in (0.95, 0.97, 0.99)]}
    retain = dg["min_in_domain_retention"]
    best = max(rep["detectores"], key=lambda k: next(
        r["deteccion_fuera_de_dominio"] for r in rep["detectores"][k]["por_retencion"] if r["retencion"] == retain))
    logit_thr = float(np.quantile(scores[best]["i_eval"], 1 - retain))
    thr = 1.0 / (1.0 + np.exp(-logit_thr))
    gate = {"detector": best, "umbral": thr, "retencion_objetivo": retain, "clasificador": clf, "vecindad": vec_p,
            "combinado": comb_p}
    rep["elegido"] = best
    rep["umbral"] = thr
    rep["criterio"] = f"mayor detección de fuera-de-dominio a {retain:.0%} de retención de ISOT val (mitad de evaluación)"
    rep["particiones"] = {k: len(v) for k, v in d.items()}
    rep["dejar_un_dominio_afuera"] = leave_one_domain_out(d, seed, dg["max_features"], retain)
    write_json(rep, paths.reports / "domain_gate_v3.json")
    return gate


def leave_one_domain_out(d: dict[str, list[str]], seed: int, max_features: int, retain: float) -> dict[str, Any]:
    """Limitación medida: ¿detecta un estilo que NO vio como negativo? (solo clasificador de texto)."""
    out: dict[str, Any] = {}
    for held, neg, ev in (("WELFake", d["l_train"], d["w_eval"]), ("LIAR", d["w_train"], d["l_eval"])):
        p = fit_text_classifier(d["i_train"], neg, seed, max_features)
        s_in, s_out = text_logit(d["i_eval"], p), text_logit(ev, p)
        out[f"sin_ver_{held}"] = {"auc": _auc(s_in, s_out), **detection_at_retention(s_in, s_out, retain)}
    return out


def save_gate(gate: dict[str, Any], path: Path | None = None) -> Path:
    """Guarda la compuerta como JSON comprimido (liviano y versionable; lo usa el quickstart)."""
    path = path or get_paths().models / GATE_FILE
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(gate, fh, ensure_ascii=False)
    return path


def load_gate(path: Path | None = None) -> dict[str, Any] | None:
    """Carga la compuerta guardada, o None si no existe."""
    path = path or get_paths().models / GATE_FILE
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def answered_report(df: pd.DataFrame, p: np.ndarray, s: np.ndarray, gate: dict[str, Any], threshold: float,
                    majority: int) -> dict[str, Any]:
    """Accuracy sobre lo que el sistema SÍ responde (en dominio) frente al trivial sobre lo mismo."""
    y = df["label"].to_numpy()
    inside = s >= gate["umbral"]
    pred = (p >= threshold).astype(int)
    return {"n": int(len(y)), "frac_fuera_de_dominio": float((~inside).mean()), "n_en_dominio": int(inside.sum()),
            "accuracy_todos_forzado": float((pred == y).mean()),
            "accuracy_en_dominio": float((pred[inside] == y[inside]).mean()) if inside.any() else None,
            "trivial_en_dominio": float((y[inside] == majority).mean()) if inside.any() else None,
            "prop_falsas_en_dominio": float(y[inside].mean()) if inside.any() else None}
