"""Entrenamiento del modelo final de la iteración 3 (cierre).

Decisiones solo con ISOT val y desarrollo externo (WELFake dev, LIAR valid). Corrige la contracción
global de la iteración 2: dentro del dominio la probabilidad es la calibrada, sin deformar; fuera del
dominio el sistema no emite veredicto (`domain.py`).
"""

from __future__ import annotations

import datetime as dt
import logging
import platform
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

from . import models as M
from . import plots
from .config import config_hash, get_paths, load_config, read_json, set_seed, sha256_file, write_json
from .corpus import isot_split, liar_split, welfake_split
from .domain import answered_report, domain_score, fit_domain_gate, load_gate, save_gate
from .explain import global_top_terms
from .invariance import TRANSFORMS_V3, flip_stats_ci
from .metrics import classification_metrics, compact, reliability_bins
from .ood import ood_features, ood_score, shrink
from .panel import fit_model, invariance_inputs_v3
from .thresholds import choose_confidence_cuts, choose_threshold, zone_summary
from .train_v2 import _calibrate_if_better

LOGGER = logging.getLogger(__name__)
ARTIFACT = "final_model_v3.joblib"
RETRIEVAL = "retrieval_v3.joblib"


def selected_spec() -> dict[str, Any]:
    """Spec del modelo final: override en config o elección del panel v3."""
    cfg, paths = load_config(), get_paths()
    name = cfg.get("final_v3", {}).get("candidate")
    specs = {c["name"]: c for c in cfg["panel_v3"]["candidates"]}
    if name:
        return specs[name]
    return read_json(paths.reports / "panel_v3.json")["seleccion"]["spec"]


def invariance_report(model: Any, profile: str, threshold: float) -> dict[str, Any]:
    """Invarianza del modelo final (calibrado): tasa de cambio con IC 95 % por transformación."""
    cfg = load_config()
    inv = invariance_inputs_v3(profile, cfg["panel"]["invariance_sample"], cfg["seed"])
    base = M.p_fake(model, inv["base"])
    rows = {k: flip_stats_ci(base, M.p_fake(model, inv[k]), threshold, cfg["seed"]) for k in TRANSFORMS_V3}
    limit = cfg["panel"]["max_invariance_flip"]
    return {"limite_cota_superior": limit, "n": len(inv["base"]), "transformaciones": rows,
            "cumple_todas": bool(all(v["ic95_sup"] <= limit for v in rows.values())),
            "no_cumplen": [k for k, v in rows.items() if v["ic95_sup"] > limit]}


def shrink_resolution(model: Any, profile: str, t: float, gate: dict[str, Any], retrieval: dict[str, Any],
                      vocab: Any) -> dict[str, Any]:
    """ECE y Brier antes (it2: contracción global) y después (it3), solo en validación y desarrollo."""
    paths = get_paths()
    a2 = joblib.load(paths.models / "final_model_v2.joblib")
    r2 = joblib.load(paths.models / "retrieval_v2.joblib")
    out: dict[str, Any] = {"nota": "Solo ISOT val, WELFake dev y LIAR valid. Nunca test ni holdouts."}
    sets = (("isot_val", lambda p: isot_split("val", p)), ("welfake_dev", lambda p: welfake_split("dev", p)),
            ("liar_valid", lambda p: liar_split("valid", p)))
    for name, load in sets:
        d2, d3 = load("v2"), load(profile)
        y = d3["label"].to_numpy()
        p2 = M.p_fake(a2["model"], d2["doc"].tolist())
        s2 = ood_score(ood_features(d2["doc"].tolist(), r2, a2["vocabulary"]), a2["ood"])
        p2c = shrink(p2, s2, a2["ood"]["prior"])
        p3 = M.p_fake(model, d3["doc"].tolist())
        inside = domain_score(d3["doc"].tolist(), gate, retrieval, vocab) >= gate["umbral"]

        def m(yy: np.ndarray, pp: np.ndarray) -> dict[str, float]:
            r = classification_metrics(yy, (pp >= t).astype(int), pp)
            return {"ece": r["ece"], "brier": r["brier"], "accuracy": r["accuracy"], "n": int(len(yy))}

        out[name] = {"it2_contraccion_global": m(y, p2c), "it3_calibrada_todos": m(y, p3),
                     "it3_calibrada_en_dominio": m(y[inside], p3[inside]) if inside.sum() >= 30 else None}
    return out


def train_final_v3(quick: bool = False) -> dict[str, Any]:
    """Entrena el modelo final, fija umbrales y compuerta, y guarda el artefacto."""
    cfg, paths = load_config(), get_paths()
    seed = set_seed()
    spec = selected_spec()
    prof = spec["profile"]
    LOGGER.info("Modelo final v3: %s (perfil %s)", spec["name"], prof)
    tr, va = isot_split("train", prof), isot_split("val", prof)
    ytr, yva = tr["label"].to_numpy(), va["label"].to_numpy()
    model = fit_model(spec, tr["doc"].tolist(), ytr)
    model, cal = _calibrate_if_better(spec, model, tr, va)
    p_va = M.p_fake(model, va["doc"].tolist())
    dc = cfg["decision"]
    thr = choose_threshold(yva, p_va, dc["threshold_grid"], dc["max_fpr"])
    t = thr["umbral"]
    conf = choose_confidence_cuts(yva, p_va, t, dc["ambiguous_low"], dc["ambiguous_high"], dc["confidence_grid"],
                                  dc["confidence_high_target_accuracy"], window=dc["confidence_window"])
    knn = M.knn(15).fit(tr["doc"].tolist(), ytr)
    retrieval = {"pipeline": M.strip_heavy_attributes(knn), "ids": tr["id"].tolist(),
                 "titles": tr["doc"].str.split("\n").str[0].str[:160].tolist(), "labels": ytr.tolist(), "profile": prof}
    vocab = frozenset(CountVectorizer(lowercase=True, min_df=cfg["tfidf"]["min_df"]).fit(tr["doc"]).vocabulary_)
    gate = load_gate() if quick else fit_domain_gate(prof, retrieval, vocab)
    if gate is None:
        raise FileNotFoundError("Falta models/domain_gate_v3.json.gz (viene versionado en el repo).")
    if not quick:
        save_gate(gate)
    prior = float(ytr.mean())
    meta = {"iteracion": 3, "modelo": spec["name"], "spec": spec, "perfil_limpieza": prof,
            "version": f"v3-{spec['name']}-{config_hash()}" + ("-quick" if quick else ""),
            "creado": dt.datetime.now().isoformat(timespec="seconds"), "semilla": seed, "config_hash": config_hash(),
            "umbral": t, "cortes_confianza": conf["cortes"], "zona_ambigua": [dc["ambiguous_low"], dc["ambiguous_high"]],
            "calibracion": cal["elegido"], "dominio": {"detector": gate["detector"], "umbral": gate["umbral"],
                                                       "retencion_objetivo": gate["retencion_objetivo"]},
            "datos": {k: {"sha256_ids": v["sha256_ids"], "n": v["total"]}
                      for k, v in read_json(paths.reports / "splits_manifest.json")["splits"].items()},
            "metricas_val": compact(classification_metrics(yva, (p_va >= t).astype(int), p_va)),
            "versiones": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
                          "pandas": pd.__version__, "scipy": scipy.__version__}}
    n_words = tr["n_words"].to_numpy()
    artifact = {"name": spec["name"], "model": M.strip_heavy_attributes(model), "profile": prof, "threshold": t,
                "ambiguous": [dc["ambiguous_low"], dc["ambiguous_high"]], "confidence_cuts": conf["cortes"],
                "vocabulary": vocab, "domain_gate": gate, "prior": prior, "iteration": 3,
                "train_words_percentiles": {str(q): float(np.percentile(n_words, q)) for q in (1, 5, 50, 95, 99)},
                "metadata": meta}
    joblib.dump(artifact, paths.models / ARTIFACT, compress=3)
    joblib.dump(retrieval, paths.models / RETRIEVAL, compress=3)
    if quick:
        LOGGER.info("Modelo rápido guardado en models/%s (val F1 falsa %.4f)", ARTIFACT, meta["metricas_val"]["f1_fake"])
        return {"modo": "quickstart", "metricas_val": meta["metricas_val"]}
    write_json(meta, paths.models / "metadata_v3.json")

    majority = int(prior >= 0.5)
    wel, lv = welfake_split("dev", prof), liar_split("valid", prof)
    answered = {}
    rel = {}
    for name, df in (("isot_val", va), ("welfake_dev", wel), ("liar_valid", lv)):
        docs = df["doc"].tolist()
        p = M.p_fake(model, docs)
        s = domain_score(docs, gate, retrieval, vocab)
        answered[name] = answered_report(df, p, s, gate, t, majority)
        rel[name] = reliability_bins(df["label"].to_numpy(), p)
    plots.reliability_plot({"ISOT val": rel["isot_val"], "WELFake dev": rel["welfake_dev"], "LIAR valid": rel["liar_valid"]},
                           f"Confiabilidad del modelo final ({spec['name']}), sin contraer",
                           paths.figures / "reliability_v3.png")
    inv = invariance_report(model, prof, t)
    write_json(inv, paths.reports / "invariance_v3.json")
    names = list(inv["transformaciones"])[::-1]
    plots.hbar_groups(names, {"Cota superior IC 95 %": [inv["transformaciones"][k]["ic95_sup"] for k in names],
                              "Tasa de cambio": [inv["transformaciones"][k]["tasa_cambio"] for k in names]},
                      "Invarianza del modelo final (límite: 2 %)", "Fracción de predicciones que cambian",
                      paths.figures / "invariance_v3.png", colors=[plots.NEUTRAL, plots.SERIES[6]])
    res = shrink_resolution(model, prof, t, gate, retrieval, vocab)
    write_json(res, paths.reports / "shrink_resolution_v3.json")
    out = {"spec": spec, "calibracion": cal, "umbral": thr | {"tabla": None}, "confianza": conf,
           "zona_val": zone_summary(yva, p_va, t, dc["ambiguous_low"], dc["ambiguous_high"]),
           "dominio_dev": answered, "top_terminos": global_top_terms(model, 25), "artefacto_sha256": sha256_file(paths.models / ARTIFACT)[:16]}
    write_json(out, paths.reports / "train_v3.json")
    return out
