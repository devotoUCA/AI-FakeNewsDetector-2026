"""Entrenamiento del modelo final de la iteración 2.

Usa el candidato elegido por el panel (`reports/panel_v2.json`) o el indicado en
`config.yaml -> final_v2.candidate`. Todo se decide con ISOT val + desarrollo externo
(WELFake-dev, LIAR-valid). Nunca toca el test de ISOT ni el holdout de WELFake.
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
from .config import config_hash, get_paths, load_config, read_json, set_seed, write_json
from .corpus import isot_split, liar_split, welfake_split
from .metrics import classification_metrics, compact, reliability_bins
from .ood import abstention_table, choose_tau, fit_ood_detector, ood_features, ood_score, shrink
from .panel import fit_model, group_folds, temporal_split
from .thresholds import choose_confidence_cuts, choose_threshold, coverage_curve, zone_summary

LOGGER = logging.getLogger(__name__)
ARTIFACT = "final_model_v2.joblib"
RETRIEVAL = "retrieval_v2.joblib"


def selected_spec() -> dict[str, Any]:
    """Spec del candidato final: override en config o elección del panel."""
    cfg, paths = load_config(), get_paths()
    override = cfg.get("final_v2", {}).get("candidate")
    specs = {c["name"]: c for c in cfg["panel"]["candidates"]}
    if override:
        return specs[override]
    panel = paths.reports / "panel_v2.json"
    if not panel.exists():
        raise FileNotFoundError("Falta reports/panel_v2.json: corré `python -m fakenews.cli panel` "
                                "o fijá final_v2.candidate en config.yaml.")
    return read_json(panel)["seleccion"]["spec"]


def _calibrate_if_better(spec: dict[str, Any], model: Any, tr: pd.DataFrame, va: pd.DataFrame
                         ) -> tuple[Any, dict[str, Any]]:
    """Para LR sin D1: elige entre {ninguna, sigmoide, isotónica} por Brier en validación."""
    p_raw = M.p_fake(model, va["doc"].tolist())
    y = va["label"].to_numpy()
    rep = {"none": classification_metrics(y, (p_raw >= 0.5).astype(int), p_raw)}
    best, best_model = "none", model
    if spec["model"] in ("lr_word", "lr_char") and not spec.get("d1"):
        from .panel import build_model

        for method in ("sigmoid", "isotonic"):
            cal = M.calibrated(build_model(spec), method).fit(tr["doc"].tolist(), tr["label"].to_numpy())
            p = M.p_fake(cal, va["doc"].tolist())
            rep[method] = classification_metrics(y, (p >= 0.5).astype(int), p)
            if rep[method]["brier"] < rep[best]["brier"]:
                best, best_model = method, cal
    summary = {k: {"brier": v["brier"], "ece": v["ece"], "log_loss": v["log_loss"]} for k, v in rep.items()}
    return best_model, {"elegido": best, "criterio": "menor Brier en ISOT val", "opciones": summary}


def shift_reliability(spec: dict[str, Any]) -> dict[str, Any]:
    """Confiabilidad (ECE y bins) del candidato final en las vistas con cambio de distribución."""
    prof = spec["profile"]
    pool = pd.concat([isot_split("train", prof), isot_split("val", prof)], ignore_index=True)
    out: dict[str, Any] = {}
    ys, ps = [], []
    for f in group_folds(pool):
        fit, ev = pool.iloc[f["train_idx"]], pool.iloc[f["eval_idx"]]
        m = fit_model(spec, fit["doc"].tolist(), fit["label"].to_numpy())
        ys.append(ev["label"].to_numpy())
        ps.append(M.p_fake(m, ev["doc"].tolist()))
    y, p = np.concatenate(ys), np.concatenate(ps)
    out["grupo"] = {"ece": classification_metrics(y, (p >= .5).astype(int), p)["ece"],
                    "bins": reliability_bins(y, p)}
    ts = temporal_split(pool)
    fit, ev = pool.iloc[ts["train_idx"]], pool.iloc[ts["eval_idx"]]
    m = fit_model(spec, fit["doc"].tolist(), fit["label"].to_numpy())
    p = M.p_fake(m, ev["doc"].tolist())
    y = ev["label"].to_numpy()
    out["tiempo"] = {"ece": classification_metrics(y, (p >= .5).astype(int), p)["ece"], "bins": reliability_bins(y, p)}
    return out


def train_final_v2(quick: bool = False) -> dict[str, Any]:
    """Entrena el modelo final v2, calibra decisión, OOD y abstención, y guarda el artefacto.

    Con ``quick=True`` (inicio rápido) no usa los externos de desarrollo: reutiliza los parámetros
    OOD ya calibrados en `models/ood_params_v2.json` (versionado) y no calcula diagnósticos.
    """
    cfg, paths = load_config(), get_paths()
    seed = set_seed()
    spec = selected_spec()
    prof = spec["profile"]
    LOGGER.info("Modelo final v2: %s (perfil %s)", spec["name"], prof)
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

    # Índice de recuperación (similares + rasgos OOD) y vocabulario, con el mismo perfil
    knn = M.knn(cfg["models"]["knn"].get("k_retrieval", 15)).fit(tr["doc"].tolist(), ytr)
    retrieval = {"pipeline": M.strip_heavy_attributes(knn), "ids": tr["id"].tolist(),
                 "titles": tr["doc"].str.split("\n").str[0].str[:160].tolist(), "labels": ytr.tolist(),
                 "profile": prof}
    vocab = frozenset(CountVectorizer(lowercase=True, min_df=cfg["tfidf"]["min_df"]).fit(tr["doc"]).vocabulary_)

    if quick:
        return _save_quick(spec, model, cal, t, conf, retrieval, vocab, tr, yva, p_va)

    # OOD: ISOT-val (dentro) vs desarrollo externo (WELFake-dev + LIAR-valid)
    wel, lv = welfake_split("dev", prof), liar_split("valid", prof)
    ext_docs = wel["doc"].tolist() + lv["doc"].tolist()
    y_ext = np.r_[wel["label"].to_numpy(), lv["label"].to_numpy()]
    f_in = ood_features(va["doc"].tolist(), retrieval, vocab)
    f_ext = ood_features(ext_docs, retrieval, vocab)
    params = fit_ood_detector(f_in, f_ext, seed)
    s_in, s_ext = ood_score(f_in, params), ood_score(f_ext, params)
    prior = float(ytr.mean())
    p_ext = M.p_fake(model, ext_docs)
    majority = int(prior >= 0.5)
    oc = cfg["ood_v2"]
    table = abstention_table(s_in, s_ext, y_ext, p_ext, yva, p_va, t, majority)
    tau = choose_tau(table, oc["min_in_domain_coverage"])
    tau_cap = float(np.quantile(s_in, oc["cap_quantile_in_domain"]))
    params.update({"tau": tau["tau"], "tau_cap": min(tau_cap, tau["tau"]), "prior": prior})

    # Efecto de la contracción sobre la calibración externa (desarrollo)
    p_ext_s = shrink(p_ext, s_ext, prior)
    p_va_s = shrink(p_va, s_in, prior)
    calib_ext = {
        "welfake_dev": {"sin_contraccion": compact(classification_metrics(wel["label"].to_numpy(), (p_ext[:len(wel)] >= t).astype(int), p_ext[:len(wel)])),
                        "con_contraccion": compact(classification_metrics(wel["label"].to_numpy(), (p_ext_s[:len(wel)] >= t).astype(int), p_ext_s[:len(wel)]))},
        "liar_valid": {"sin_contraccion": compact(classification_metrics(lv["label"].to_numpy(), (p_ext[len(wel):] >= t).astype(int), p_ext[len(wel):])),
                       "con_contraccion": compact(classification_metrics(lv["label"].to_numpy(), (p_ext_s[len(wel):] >= t).astype(int), p_ext_s[len(wel):]))},
        "isot_val": {"sin_contraccion": compact(classification_metrics(yva, (p_va >= t).astype(int), p_va)),
                     "con_contraccion": compact(classification_metrics(yva, (p_va_s >= t).astype(int), p_va_s))},
    }
    rel = {"ISOT val": reliability_bins(yva, p_va),
           "WELFake-dev (sin contraer)": reliability_bins(wel["label"].to_numpy(), p_ext[:len(wel)]),
           "WELFake-dev (contraída)": reliability_bins(wel["label"].to_numpy(), p_ext_s[:len(wel)])}
    shift = shift_reliability(spec)
    rel["Grupo (folds)"] = shift["grupo"]["bins"]
    rel["Tiempo"] = shift["tiempo"]["bins"]
    plots.reliability_plot(rel, f"Confiabilidad del modelo final v2 ({spec['name']})",
                           paths.figures / "reliability_shift_v2.png")
    # Curva cobertura vs accuracy en desarrollo externo (variando tau)
    plots.line_plot({"Externo dev: accuracy respondidos": ([r["cobertura_ext"] for r in table], [r["accuracy_ext_respondidos"] for r in table]),
                     "Externo dev: trivial respondidos": ([r["cobertura_ext"] for r in table], [r["trivial_ext_respondidos"] for r in table]),
                     "ISOT val: accuracy respondidos": ([r["cobertura_in"] for r in table], [r["accuracy_in_respondidos"] for r in table])},
                    "Abstención por OOD: cobertura vs accuracy (desarrollo)", "Cobertura", "Accuracy",
                    paths.figures / "abstention_coverage_v2.png",
                    colors=[plots.SERIES[6], plots.NEUTRAL, plots.SERIES[2]])

    meta = {"iteracion": 2, "modelo": spec["name"], "spec": spec, "perfil_limpieza": prof,
            "version": f"v2-{spec['name']}-{config_hash()}", "creado": dt.datetime.now().isoformat(timespec="seconds"),
            "semilla": seed, "config_hash": config_hash(), "umbral": t, "cortes_confianza": conf["cortes"],
            "zona_ambigua": [dc["ambiguous_low"], dc["ambiguous_high"]], "calibracion": cal["elegido"],
            "ood": {k: v for k, v in params.items() if k in ("tau", "tau_cap", "prior")},
            "datos": {k: {"sha256_ids": v["sha256_ids"], "n": v["total"]}
                      for k, v in read_json(paths.reports / "splits_manifest.json")["splits"].items()},
            "metricas_val": compact(classification_metrics(yva, (p_va >= t).astype(int), p_va)),
            "versiones": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                          "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}}
    n_words = tr["n_words"].to_numpy()
    artifact = {"name": spec["name"], "model": M.strip_heavy_attributes(model), "profile": prof, "threshold": t,
                "ambiguous": [dc["ambiguous_low"], dc["ambiguous_high"]], "confidence_cuts": conf["cortes"],
                "vocabulary": vocab, "ood": params, "iteration": 2,
                "train_words_percentiles": {str(q): float(np.percentile(n_words, q)) for q in (1, 5, 50, 95, 99)},
                "metadata": meta}
    joblib.dump(artifact, paths.models / ARTIFACT, compress=3)
    joblib.dump(retrieval, paths.models / RETRIEVAL, compress=3)
    write_json(meta, paths.models / "metadata_v2.json")
    write_json({"spec": spec, **params}, paths.models / "ood_params_v2.json")
    out = {"spec": spec, "calibracion": cal, "umbral": thr | {"tabla": None}, "confianza": conf,
           "zona_val": zone_summary(yva, p_va, t, dc["ambiguous_low"], dc["ambiguous_high"]),
           "cobertura_val": coverage_curve(yva, p_va, t),
           "ood": {"parametros": params, "tabla_abstencion_dev": table, "eleccion_tau": tau,
                   "rasgos_media_in": dict(zip(params["features"], f_in.mean(0).round(4).tolist())),
                   "rasgos_media_ext": dict(zip(params["features"], f_ext.mean(0).round(4).tolist())),
                   "frac_in_abstiene": float((s_in >= params["tau"]).mean()),
                   "frac_welfake_dev_abstiene": float((s_ext[:len(wel)] >= params["tau"]).mean()),
                   "frac_liar_valid_abstiene": float((s_ext[len(wel):] >= params["tau"]).mean())},
           "calibracion_externa_dev": calib_ext, "ece_vistas_con_cambio": {k: v["ece"] for k, v in shift.items()}}
    write_json(out, paths.reports / "train_v2.json")
    LOGGER.info("Final v2 guardado: umbral %.2f, tau OOD %.2f", t, params["tau"])
    return out


def _save_quick(spec: dict[str, Any], model: Any, cal: dict[str, Any], t: float, conf: dict[str, Any],
                retrieval: dict[str, Any], vocab: frozenset[str], tr: pd.DataFrame, yva: np.ndarray,
                p_va: np.ndarray) -> dict[str, Any]:
    """Guarda el artefacto del inicio rápido reutilizando los parámetros OOD versionados."""
    cfg, paths = load_config(), get_paths()
    dc = cfg["decision"]
    ood_path = paths.models / "ood_params_v2.json"
    params = None
    if ood_path.exists():
        saved = read_json(ood_path)
        if saved.get("spec", {}).get("name") == spec["name"]:
            params = {k: v for k, v in saved.items() if k != "spec"}
    if params is None:
        LOGGER.warning("No hay parámetros OOD para %s: el modelo rápido usará solo las reglas de entrada.", spec["name"])
    meta = {"iteracion": 2, "modelo": spec["name"], "spec": spec, "perfil_limpieza": spec["profile"],
            "version": f"v2-{spec['name']}-{config_hash()}-quick", "creado": dt.datetime.now().isoformat(timespec="seconds"),
            "semilla": cfg["seed"], "config_hash": config_hash(), "umbral": t, "cortes_confianza": conf["cortes"],
            "calibracion": cal["elegido"], "modo": "quickstart",
            "metricas_val": compact(classification_metrics(yva, (p_va >= t).astype(int), p_va)),
            "versiones": {"python": platform.python_version(), "sklearn": sklearn.__version__}}
    n_words = tr["n_words"].to_numpy()
    artifact = {"name": spec["name"], "model": M.strip_heavy_attributes(model), "profile": spec["profile"],
                "threshold": t, "ambiguous": [dc["ambiguous_low"], dc["ambiguous_high"]],
                "confidence_cuts": conf["cortes"], "vocabulary": vocab, "ood": params, "iteration": 2,
                "train_words_percentiles": {str(q): float(np.percentile(n_words, q)) for q in (1, 5, 50, 95, 99)},
                "metadata": meta}
    joblib.dump(artifact, paths.models / ARTIFACT, compress=3)
    joblib.dump(retrieval, paths.models / RETRIEVAL, compress=3)
    LOGGER.info("Modelo rápido guardado en models/%s (val F1 falsa %.4f)", ARTIFACT, meta["metricas_val"]["f1_fake"])
    return {"modo": "quickstart", "metricas_val": meta["metricas_val"], "ood": bool(params)}
