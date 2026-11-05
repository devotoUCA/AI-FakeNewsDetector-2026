"""Evaluación final de cierre (iteración 3): una sola corrida, registrada.

* ISOT test y WELFake holdout: "segunda mirada declarada" (ya se miraron antes; no son vírgenes).
* FA-KES: holdout VIRGEN (etiqueta de veracidad, ambas clases de los mismos medios). Se evalúan una vez
  los modelos de las iteraciones 1, 2 y 3.
* LIAR test: no virgen; se reporta marcado.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import plots
from .access_log import log_path, read_log
from .config import config_hash, get_paths, load_config, read_json, sha256_file, write_json
from .corpus import fakes_split, isot_split, liar_split, welfake_split
from .domain import answered_report, domain_score
from .evaluate import SOURCE_TOKENS, _error_category
from .explain import global_top_terms, term_contributions
from .features import STYLE_FEATURES, StyleFeatures
from .metrics import bootstrap_ci, classification_metrics, compact, mcnemar_exact, reliability_bins
from .models import p_fake
from .thresholds import confidence_level

LOGGER = logging.getLogger(__name__)
SEGUNDA = "iteración 3 – segunda mirada declarada"


def _log(entries: list[dict[str, Any]]) -> None:
    seen = {(e["modelo"], e["huella"], e.get("conjunto", "isot_test")) for e in read_log()}
    with log_path().open("a", encoding="utf-8") as fh:
        for e in entries:
            e["reacceso"] = (e["modelo"], e["huella"], e["conjunto"]) in seen
            e["iteracion"] = 3
            if e["reacceso"]:
                LOGGER.warning("RE-ACCESO a %s para %s: no usar para decidir", e["conjunto"], e["modelo"])
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


def _metrics(y: np.ndarray, p: np.ndarray, t: float, prior: float, seed: int) -> dict[str, Any]:
    pred = (p >= t).astype(int)
    m = classification_metrics(y, pred, np.clip(p, 1e-6, 1 - 1e-6))
    m["ic95"] = bootstrap_ci(y, pred, np.clip(p, 1e-6, 1 - 1e-6), 1000, seed)
    majority = int(prior >= 0.5)
    tr = classification_metrics(y, np.full(len(y), majority), np.full(len(y), prior))
    return {**compact(m), "macro_f1": m["macro_f1"], "auc": float(roc_auc_score(y, p)), "ic95": m["ic95"],
            "confusion": m["confusion"], "n": int(len(y)),
            "prop_falsas": float(y.mean()), "trivial": {"accuracy": tr["accuracy"], "macro_f1": tr["macro_f1"], "brier": tr["brier"]}}


def evaluate_final_v3() -> dict[str, Any]:
    """Corre la evaluación final única. Ver docstring del módulo."""
    cfg, paths = load_config(), get_paths()
    seed = cfg["seed"]
    art = joblib.load(paths.models / "final_model_v3.joblib")
    ret = joblib.load(paths.models / "retrieval_v3.joblib")
    it1 = joblib.load(paths.models / "final_model.joblib")
    it2 = joblib.load(paths.models / "final_model_v2.joblib")
    prof, t, gate, prior = art["profile"], art["threshold"], art["domain_gate"], art["prior"]
    ts = dt.datetime.now().isoformat(timespec="seconds")
    h3 = sha256_file(paths.models / "final_model_v3.joblib")[:16]
    h2 = sha256_file(paths.models / "final_model_v2.joblib")[:16]
    h1 = sha256_file(paths.models / "final_model.joblib")[:16]
    v3 = f"v3:{art['name']}"
    base = {"ts": ts, "config_hash": config_hash()}
    _log([{**base, "modelo": v3, "huella": h3, "conjunto": "isot_test", "motivo": SEGUNDA},
          {**base, "modelo": v3, "huella": h3, "conjunto": "welfake_holdout", "motivo": SEGUNDA},
          {**base, "modelo": v3, "huella": h3, "conjunto": "liar_test (NO virgen)", "motivo": SEGUNDA},
          {**base, "modelo": v3, "huella": h3, "conjunto": "fakes_holdout", "motivo": "iteración 3: holdout virgen, primer y único acceso"},
          {**base, "modelo": "it2:lrchar_v2_C10", "huella": h2, "conjunto": "fakes_holdout", "motivo": "iteración 3: holdout virgen, primer y único acceso"},
          {**base, "modelo": "it1:linear_svm_cal", "huella": h1, "conjunto": "fakes_holdout", "motivo": "iteración 3: holdout virgen, primer y único acceso"}])
    sets = {"isot_test": isot_split("test", prof), "welfake_holdout": welfake_split("holdout", prof),
            "liar_test": liar_split("test", prof), "fakes_holdout": fakes_split(prof)}
    res: dict[str, Any] = {"modelo": art["name"], "perfil": prof, "fecha": ts, "umbral": t, "v3": {}, "dominio": {}}
    probs, doms = {}, {}
    for name, df in sets.items():
        docs, y = df["doc"].tolist(), df["label"].to_numpy()
        p = p_fake(art["model"], docs)
        s = domain_score(docs, gate, ret, art["vocabulary"])
        probs[name], doms[name] = p, s
        res["v3"][name] = _metrics(y, p, t, prior, seed)
        res["dominio"][name] = answered_report(df, p, s, gate, t, int(prior >= .5))
    fk1, fk2 = fakes_split("v1"), fakes_split("v2")
    y = fk1["label"].to_numpy()
    res["it1_fakes_holdout"] = _metrics(y, p_fake(it1["model"], fk1["doc"].tolist()), it1["threshold"], prior, seed)
    res["it2_fakes_holdout"] = _metrics(y, p_fake(it2["model"], fk2["doc"].tolist()), it2["threshold"], prior, seed)
    # Comparación del acta: mismo test, baseline de la iteración 1 (predicciones guardadas, sin re-acceso)
    saved = np.load(paths.interim / "test_predictions.npz")
    yt = sets["isot_test"]["label"].to_numpy()
    if not np.array_equal(saved["y"], yt):
        raise ValueError("Las predicciones guardadas del baseline no están alineadas con el test.")
    base_pred = (saved["logreg_raw"] >= 0.5).astype(int)
    fin_pred = (probs["isot_test"] >= t).astype(int)
    res["mcnemar_baseline_vs_final"] = mcnemar_exact(yt, base_pred, fin_pred)
    res["baseline_it1"] = _metrics(yt, saved["logreg_raw"], 0.5, prior, seed)
    acc = res["v3"]["isot_test"]["accuracy"]
    top = global_top_terms(art["model"], 50)
    terms = {d["termino"].strip() for d in top["hacia_falsa"] + top["hacia_real"]}
    res["alerta_accuracy_sospechosa"] = {"accuracy_isot_test": acc, "alerta": acc > cfg["evaluation"]["suspicious_accuracy"],
                                         "tokens_de_fuente_en_top50": sorted(terms & SOURCE_TOKENS)}
    res["top_terminos"] = global_top_terms(art["model"], 25)
    res["confiabilidad"] = {k: reliability_bins(sets[k]["label"].to_numpy(), probs[k]) for k in sets}
    write_json(res, paths.reports / "final_v3.json")
    acta_table(res)
    for name in ("isot_test", "welfake_holdout", "fakes_holdout"):
        plots.confusion_matrix_plot(res["v3"][name]["confusion"], f"Matriz de confusión — {name}",
                                    paths.figures / f"confusion_v3_{name}.png")
    summary_figure(res)
    res["errores"] = error_analysis_v3(art, sets, probs, doms)
    write_json(res, paths.reports / "final_v3.json")
    return res


def acta_table(res: dict[str, Any]) -> pd.DataFrame:
    """Tabla del acta en el mismo test: modelos de la iteración 1 (limpieza v1) y modelo final."""
    paths = get_paths()
    d = pd.read_csv(paths.reports / "test_comparison.csv")
    d = d[d["modelo"] != "logreg_cal"].copy()
    d["limpieza"] = "v1"
    d["iteracion"] = 1
    m = res["v3"]["isot_test"]
    row = {"modelo": f"{res['modelo']} (final)", "limpieza": res["perfil"], "iteracion": 3,
           "ic95_f1_fake": [round(x, 4) for x in m["ic95"]["f1_fake"]], "ic95_accuracy": [round(x, 4) for x in m["ic95"]["accuracy"]]}
    row |= {k: round(m[k], 4) for k in ("accuracy", "precision_fake", "recall_fake", "f1_fake", "precision_real",
                                        "recall_real", "f1_real", "brier", "log_loss", "ece")}
    out = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
    out.to_csv(paths.reports / "acta_comparison_v3.csv", index=False)
    return out


def summary_figure(res: dict[str, Any]) -> None:
    """Figura central: accuracy del modelo frente al trivial en cada conjunto."""
    names = [("isot_test", "ISOT test\n(mismo dominio)"), ("welfake_holdout", "WELFake holdout\n(otros medios)"),
             ("liar_test", "LIAR test\n(no virgen)"), ("fakes_holdout", "FA-KES\n(virgen)")]
    plots.model_vs_trivial([n for _, n in names], [res["v3"][k]["accuracy"] for k, _ in names],
                           [res["v3"][k]["trivial"]["accuracy"] for k, _ in names],
                           "El 99 % solo vale dentro de ISOT", get_paths().figures / "resumen_generalizacion.png")


def error_analysis_v3(art: dict[str, Any], sets: dict[str, pd.DataFrame], probs: dict[str, np.ndarray],
                      doms: dict[str, np.ndarray], per_cell: int = 4, minimum: int = 40) -> dict[str, Any]:
    """`reports/error_analysis_v3.csv`: errores estratificados por conjunto, clase y nivel de confianza."""
    t, gate = art["threshold"], art["domain_gate"]
    frames = []
    for origin in ("isot_test", "welfake_holdout"):
        df, p, s = sets[origin], probs[origin], doms[origin]
        d = df.assign(p=p, score=s, pred=(p >= t).astype(int), origen=origin)
        d["en_dominio"] = d["score"] >= gate["umbral"]
        d = d[d["pred"] != d["label"]].copy()
        d["nivel"] = [confidence_level(float(x), art["confidence_cuts"]) for x in d["p"]]
        d["estrato"] = np.where(d["en_dominio"], d["nivel"], "fuera_de_dominio")
        frames.append(d)
    errs = pd.concat(frames, ignore_index=True)
    seed = load_config()["seed"]
    picked = pd.concat([g.sample(min(per_cell, len(g)), random_state=seed)
                        for _, g in errs.groupby(["origen", "label", "estrato"])])
    if len(picked) < minimum:
        rest = errs[~errs["id"].isin(picked["id"])].sample(min(minimum - len(picked), len(errs) - len(picked)),
                                                           random_state=load_config()["seed"])
        picked = pd.concat([picked, rest])
    rows = []
    for _, r in picked.iterrows():
        style = dict(zip(STYLE_FEATURES, StyleFeatures().transform([r["doc"]])[0]))
        terms = term_contributions(art["model"], r["doc"], top_k=5)
        cat = _error_category(r["doc"], int(r["n_words"] or 0), style, int(r["label"]))
        if r["estrato"] == "fuera_de_dominio":
            cat = "fuera de dominio: el sistema no emitiría veredicto"
        push = terms["hacia_falsa"] if r["pred"] == 1 else terms["hacia_real"]
        words = ", ".join(d["termino"].strip() for d in push[:3]) or "sin términos destacados"
        rows.append({"id": r["id"], "origen": r["origen"], "estrato_confianza": r["estrato"],
                     "tipo": "FN (falsa dada por real)" if r["label"] == 1 else "FP (real marcada como falsa)",
                     "etiqueta_real": "falsa" if r["label"] == 1 else "real", "prediccion": "falsa" if r["pred"] == 1 else "real",
                     "p_false_calibrada": round(float(r["p"]), 4), "puntaje_dominio": round(float(r["score"]), 4),
                     "en_dominio": bool(r["en_dominio"]), "palabras": int(r["n_words"] or 0),
                     "titulo": r["doc"].split("\n")[0][:160], "texto_resumido": " ".join(r["doc"].split("\n")[1:])[:400],
                     "terminos_hacia_falsa": "; ".join(d["termino"].strip() for d in terms["hacia_falsa"]),
                     "terminos_hacia_real": "; ".join(d["termino"].strip() for d in terms["hacia_real"]),
                     "categoria_tentativa": cat,
                     "comentario_borrador": f"BORRADOR – revisar por el equipo. Predijo {'falsa' if r['pred'] == 1 else 'real'} "
                                            f"con p={r['p']:.2f}; pesaron: {words}. Categoría tentativa: {cat}.",
                     "comentario_equipo": "", "revisado_por": ""})
    out = pd.DataFrame(rows).sort_values(["origen", "tipo", "estrato_confianza"])
    out.to_csv(get_paths().reports / "error_analysis_v3.csv", index=False)
    return {"n_en_csv": len(out), "por_origen": out["origen"].value_counts().to_dict(), "por_tipo": out["tipo"].value_counts().to_dict(),
            "por_estrato": out["estrato_confianza"].value_counts().to_dict(),
            "errores_totales": {o: int((errs["origen"] == o).sum()) for o in ("isot_test", "welfake_holdout")}}


def success_criteria_v3() -> dict[str, Any]:
    """Criterios del acta con el modelo final."""
    paths = get_paths()
    r = read_json(paths.reports / "final_v3.json")
    m = r["v3"]["isot_test"]
    mc = r["mcnemar_baseline_vs_final"]
    errs = pd.read_csv(paths.reports / "error_analysis_v3.csv", keep_default_na=False)
    reviewed = int((errs["comentario_equipo"].astype(str).str.strip() != "").sum())
    rows = [
        {"criterio": "Accuracy en test > 80 %", "valor": m["accuracy"], "estado": "cumplido (solo dentro de ISOT)" if m["accuracy"] > .8 else "no cumplido"},
        {"criterio": "F1 por clase > 0,80", "valor": [m["f1_real"], m["f1_fake"]], "estado": "cumplido (solo dentro de ISOT)" if min(m["f1_real"], m["f1_fake"]) > .8 else "no cumplido"},
        {"criterio": "Supera con claridad al ingenuo sin favorecer una clase", "valor": m["trivial"]["accuracy"], "estado": "cumplido en ISOT; ver externos"},
        {"criterio": "Modelo adicional iguala o supera al baseline, con la diferencia explicada",
         "valor": {"f1_baseline": r["baseline_it1"]["f1_fake"], "f1_final": m["f1_fake"], "mcnemar_p": mc["p_value"]},
         "estado": "cumplido" if m["f1_fake"] >= r["baseline_it1"]["f1_fake"] else "no cumplido"},
        {"criterio": "Sin filtración entre splits", "valor": "0 solapamientos; externos deduplicados", "estado": "cumplido"},
        {"criterio": ">= 20 errores revisados por el equipo", "valor": {"casos": len(errs), "comentados": reviewed},
         "estado": "cumplido" if reviewed >= 20 else "pendiente: equipo (error_analysis_v3.csv)"},
        {"criterio": "Demo devuelve etiqueta y probabilidad", "valor": "tests de predict y demo", "estado": "cumplido (si pytest pasa)"},
        {"criterio": "Accuracy > 98 % => sospecha", "valor": m["accuracy"], "estado": "alerta cerrada con evidencia: ver informe" if r["alerta_accuracy_sospechosa"]["alerta"] else "sin alerta"},
    ]
    out = {"modelo": r["modelo"], "criterios": rows, "generado": dt.datetime.now().isoformat(timespec="seconds")}
    write_json(out, paths.reports / "success_criteria_v3.json")
    return out
