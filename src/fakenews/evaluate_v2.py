"""Evaluación final de la iteración 2 (una sola vez, registrada en el log de accesos).

* ISOT test: modelo final v2 (forzado y con abstención).
* Holdout externo final: WELFake-holdout (virgen) para el final v2 y, como referencia, el
  productivo de la iteración 1. Una evaluación por modelo, registrada.
* LIAR test: NO es virgen (se miró completo en la iteración 1); se reporta marcado como tal.
También genera `reports/error_analysis_v2.csv` y la comparación iteración 1 vs 2.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import plots
from .cleaning import clean_text
from .config import config_hash, get_paths, load_config, read_json, sha256_file, write_json
from .corpus import isot_split, liar_split, welfake_split
from .evaluate import SOURCE_TOKENS, _draft_comment, _error_category
from .explain import global_top_terms, term_contributions
from .features import STYLE_FEATURES, StyleFeatures
from .metrics import bootstrap_ci, classification_metrics, compact, reliability_bins
from .models import p_fake
from .ood import ood_features, ood_score, shrink
from .thresholds import accuracy_by_level, zone_summary

LOGGER = logging.getLogger(__name__)
REASON = "iteración 2: evaluación final única tras selección con panel (val + desarrollo externo)"


def _log_access(entries: list[dict[str, Any]]) -> None:
    path = get_paths().reports / "test_access_log.jsonl"
    prev = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x] if path.exists() else []
    seen = {(e["modelo"], e["huella"], e.get("conjunto", "isot_test")) for e in prev}
    with path.open("a", encoding="utf-8") as fh:
        for e in entries:
            e["reacceso"] = (e["modelo"], e["huella"], e["conjunto"]) in seen
            e["motivo"] = REASON
            e["iteracion"] = 2
            if e["reacceso"]:
                LOGGER.warning("RE-ACCESO a %s para %s: no usar para decidir", e["conjunto"], e["modelo"])
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


def _system(art: dict[str, Any], retrieval: dict[str, Any] | None, docs: list[str]) -> dict[str, np.ndarray]:
    """Probabilidad cruda, puntaje OOD, probabilidad contraída y máscara de abstención OOD."""
    p = p_fake(art["model"], docs)
    if art.get("ood") and retrieval is not None:
        s = ood_score(ood_features(docs, retrieval, art["vocabulary"]), art["ood"])
        return {"p_raw": p, "s": s, "p": shrink(p, s, art["ood"]["prior"]), "abst": s >= art["ood"]["tau"]}
    return {"p_raw": p, "s": np.zeros(len(p)), "p": p, "abst": np.zeros(len(p), dtype=bool)}


def _report(y: np.ndarray, sysv: dict[str, np.ndarray], art: dict[str, Any], prior_train: float,
            seed: int) -> dict[str, Any]:
    t = art["threshold"]
    lo, hi = art["ambiguous"]
    p_raw, p, abst = sysv["p_raw"], sysv["p"], sysv["abst"]
    forced = classification_metrics(y, (p_raw >= t).astype(int), p_raw)
    forced["ic95"] = bootstrap_ci(y, (p_raw >= t).astype(int), p_raw, 500, seed)
    contr = classification_metrics(y, (p >= t).astype(int), p)
    majority = int(prior_train >= 0.5)
    amb = (p >= lo) & (p <= hi)
    answered = ~abst & ~amb
    pred = (p >= t).astype(int)
    triv = classification_metrics(y, np.full(len(y), majority), np.full(len(y), prior_train))
    return {
        "n": int(len(y)), "prop_falsas": float(y.mean()),
        "forzado": compact(forced) | {"macro_f1": forced["macro_f1"], "ic95": forced["ic95"],
                                      "confusion": forced["confusion"]},
        "contraido": compact(contr) | {"macro_f1": contr["macro_f1"]},
        "trivial": compact(triv) | {"macro_f1": triv["macro_f1"]},
        "sistema": {"cobertura": float(answered.mean()), "abstiene_por_ood": float(abst.mean()),
                    "zona_ambigua": float((amb & ~abst).mean()),
                    "accuracy_respondidos": float((pred[answered] == y[answered]).mean()) if answered.any() else None,
                    "trivial_respondidos": float((y[answered] == majority).mean()) if answered.any() else None,
                    "n_respondidos": int(answered.sum())},
        "por_nivel_confianza_sin_ood": accuracy_by_level(y, p, t, art["confidence_cuts"]),
        "confiabilidad_contraida": reliability_bins(y, p),
    }


def evaluate_final_v2() -> dict[str, Any]:
    """Evaluación única de la iteración 2 (ver docstring del módulo)."""
    cfg, paths = load_config(), get_paths()
    seed = cfg["seed"]
    art = joblib.load(paths.models / "final_model_v2.joblib")
    ret = joblib.load(paths.models / "retrieval_v2.joblib")
    it1 = joblib.load(paths.models / "final_model.joblib")
    prof = art["profile"]
    prior = art["ood"]["prior"] if art.get("ood") else 0.453
    ts = dt.datetime.now().isoformat(timespec="seconds")
    h2 = sha256_file(paths.models / "final_model_v2.joblib")[:16]
    h1 = sha256_file(paths.models / "final_model.joblib")[:16]
    wel_manifest = read_json(paths.reports / "welfake_manifest_v2.json")
    isot_manifest = read_json(paths.reports / "splits_manifest.json")
    _log_access([
        {"ts": ts, "modelo": f"v2:{art['name']}", "huella": h2, "conjunto": "isot_test",
         "config_hash": config_hash(), "test_sha256_ids": isot_manifest["splits"]["test"]["sha256_ids"]},
        {"ts": ts, "modelo": f"v2:{art['name']}", "huella": h2, "conjunto": "welfake_holdout",
         "config_hash": config_hash(), "test_sha256_ids": wel_manifest["splits"]["holdout"]["sha256_ids"]},
        {"ts": ts, "modelo": "it1:linear_svm_cal", "huella": h1, "conjunto": "welfake_holdout",
         "config_hash": config_hash(), "test_sha256_ids": wel_manifest["splits"]["holdout"]["sha256_ids"]},
        {"ts": ts, "modelo": f"v2:{art['name']}", "huella": h2, "conjunto": "liar_test (NO virgen)",
         "config_hash": config_hash(), "test_sha256_ids": "liar_oficial_test"},
        {"ts": ts, "modelo": "it1:linear_svm_cal", "huella": h1, "conjunto": "liar_test (NO virgen)",
         "config_hash": config_hash(), "test_sha256_ids": "liar_oficial_test"},
    ])
    test = isot_split("test", prof)
    wel = welfake_split("holdout", prof)
    wel1 = welfake_split("holdout", "v1")
    lt, lt1 = liar_split("test", prof), liar_split("test", "v1")
    res: dict[str, Any] = {"modelo_v2": art["name"], "perfil": prof, "fecha": ts}
    s_test = _system(art, ret, test["doc"].tolist())
    s_wel = _system(art, ret, wel["doc"].tolist())
    s_lt = _system(art, ret, lt["doc"].tolist())
    res["v2"] = {"isot_test": _report(test["label"].to_numpy(), s_test, art, prior, seed),
                 "welfake_holdout": _report(wel["label"].to_numpy(), s_wel, art, prior, seed),
                 "liar_test_no_virgen": _report(lt["label"].to_numpy(), s_lt, art, prior, seed)}
    it1_ret = None
    res["it1"] = {"welfake_holdout": _report(wel1["label"].to_numpy(), _system(it1, it1_ret, wel1["doc"].tolist()), it1, prior, seed),
                  "liar_test_no_virgen": _report(lt1["label"].to_numpy(), _system(it1, it1_ret, lt1["doc"].tolist()), it1, prior, seed),
                  "isot_test_iteracion1": compact(read_json(paths.reports / "test_results.json")["modelos"]["linear_svm_cal"])}
    acc = res["v2"]["isot_test"]["forzado"]["accuracy"]
    top = global_top_terms(art["model"], 50)
    terms = [d["termino"] for d in top["hacia_falsa"] + top["hacia_real"]]
    res["alerta_accuracy_sospechosa"] = {"accuracy_isot_test": acc, "alerta": acc > cfg["evaluation"]["suspicious_accuracy"],
                                         "tokens_de_fuente_en_top50": sorted(set(terms) & SOURCE_TOKENS)}
    res["top_terminos_v2"] = global_top_terms(art["model"], 25)
    write_json(res, paths.reports / "final_v2.json")
    for name, sv, df in (("isot_test", s_test, test), ("welfake_holdout", s_wel, wel)):
        plots.confusion_matrix_plot(res["v2"][name]["forzado"]["confusion"],
                                    f"Matriz de confusión v2 — {name}", paths.figures / f"confusion_v2_{name}.png")
    plots.reliability_plot({"ISOT test": res["v2"]["isot_test"]["confiabilidad_contraida"],
                            "WELFake holdout (contraída)": res["v2"]["welfake_holdout"]["confiabilidad_contraida"]},
                           "Confiabilidad v2 en evaluación final", paths.figures / "reliability_final_v2.png")
    res["errores"] = error_analysis_v2(art, test, s_test, wel, s_wel)
    write_json(res, paths.reports / "final_v2.json")
    comparison_table(res)
    return res


def error_analysis_v2(art: dict[str, Any], test: pd.DataFrame, s_test: dict[str, np.ndarray],
                      wel: pd.DataFrame, s_wel: dict[str, np.ndarray], per_group: int = 10) -> dict[str, Any]:
    """`reports/error_analysis_v2.csv`: >= 30 errores del modelo final (FP/FN balanceados, de ISOT test y
    del holdout externo, con alta confianza y casos OOD). No toca el archivo de la iteración 1."""
    t = art["threshold"]
    frames = []
    for origin, df, sv in (("isot_test", test, s_test), ("welfake_holdout", wel, s_wel)):
        d = df.assign(p_false=sv["p"], p_false_raw=sv["p_raw"], ood=sv["s"], abstiene_ood=sv["abst"],
                      pred=(sv["p_raw"] >= t).astype(int), origen=origin)
        frames.append(d[d["pred"] != d["label"]])
    errs = pd.concat(frames, ignore_index=True)
    errs["confianza_raw"] = np.maximum(errs["p_false_raw"], 1 - errs["p_false_raw"])
    chosen = []
    for origin in ("isot_test", "welfake_holdout"):
        for lab in (0, 1):
            sub = errs[(errs["origen"] == origin) & (errs["label"] == lab)].sort_values("confianza_raw", ascending=False)
            chosen.append(sub.head(per_group))
    sel = pd.concat(chosen).drop_duplicates("id")
    rows = []
    for _, r in sel.iterrows():
        style = dict(zip(STYLE_FEATURES, StyleFeatures().transform([r["doc"]])[0]))
        terms = term_contributions(art["model"], r["doc"], top_k=5)
        cat = _error_category(r["doc"], int(r.get("n_words", len(r["doc"].split())) or 0), style, int(r["label"]))
        if r["abstiene_ood"]:
            cat = "fuera de distribución (el sistema se abstendría)"
        rr = r.copy()
        rr["p_false"] = r["p_false_raw"]
        rows.append({"id": r["id"], "origen": r["origen"],
                     "tipo": "FN (falsa dada por real)" if r["label"] == 1 else "FP (real marcada como falsa)",
                     "etiqueta_real": "falsa" if r["label"] == 1 else "real",
                     "prediccion_forzada": "falsa" if r["pred"] == 1 else "real",
                     "p_false_cruda": round(float(r["p_false_raw"]), 4), "p_false_contraida": round(float(r["p_false"]), 4),
                     "puntaje_ood": round(float(r["ood"]), 4), "abstiene_por_ood": bool(r["abstiene_ood"]),
                     "alta_confianza_cruda": bool(r["confianza_raw"] >= 0.8),
                     "titulo": r["doc"].split("\n")[0][:160], "texto_resumido": " ".join(r["doc"].split("\n")[1:])[:400],
                     "terminos_hacia_falsa": "; ".join(d["termino"] for d in terms["hacia_falsa"]),
                     "terminos_hacia_real": "; ".join(d["termino"] for d in terms["hacia_real"]),
                     "categoria_tentativa": cat,
                     "comentario_borrador": f"BORRADOR – revisar por el equipo. {_draft_comment(rr, cat, terms)}",
                     "comentario_equipo": "", "revisado_por": ""})
    out = pd.DataFrame(rows)
    out.to_csv(get_paths().reports / "error_analysis_v2.csv", index=False)
    return {"n_en_csv": len(out), "por_origen": out["origen"].value_counts().to_dict(),
            "por_tipo": out["tipo"].value_counts().to_dict(), "alta_confianza": int(out["alta_confianza_cruda"].sum()),
            "abstiene_ood": int(out["abstiene_por_ood"].sum()),
            "por_categoria": out["categoria_tentativa"].value_counts().to_dict()}


def comparison_table(res: dict[str, Any]) -> pd.DataFrame:
    """Tabla iteración 1 vs 2 (panel + evaluación final) -> reports/comparison_it1_vs_it2.csv."""
    paths = get_paths()
    panel = pd.read_csv(paths.reports / "panel_v2.csv").set_index("candidato")
    r1, r2 = panel.loc["it1_svm_v1"], panel.loc[res["modelo_v2"]]
    rows = []
    for col, label in (("in_f1_macro", "Panel · in-domain ISOT val (F1 macro)"),
                       ("grupo_f1_macro", "Panel · cambio de grupo/subject (F1 macro)"),
                       ("tiempo_f1_macro", "Panel · cambio temporal (F1 macro)"),
                       ("welfake_dev_f1_macro", "Panel · WELFake-dev (F1 macro)"),
                       ("liar_valid_f1_macro", "Panel · LIAR-valid (F1 macro)"),
                       ("invarianza_max_cambio", "Panel · invarianza: máx. tasa de cambio"),
                       ("puntaje_robusto", "Panel · puntaje robusto")):
        rows.append({"vista": label, "iteracion_1": float(r1[col]), "iteracion_2": float(r2[col])})
    v1, v2 = res["it1"], res["v2"]
    rows += [
        {"vista": "Final · ISOT test accuracy (forzado)", "iteracion_1": v1["isot_test_iteracion1"]["accuracy"],
         "iteracion_2": v2["isot_test"]["forzado"]["accuracy"]},
        {"vista": "Final · ISOT test F1 falsa (forzado)", "iteracion_1": v1["isot_test_iteracion1"]["f1_fake"],
         "iteracion_2": v2["isot_test"]["forzado"]["f1_fake"]},
        {"vista": "Final · WELFake holdout accuracy (forzado)", "iteracion_1": v1["welfake_holdout"]["forzado"]["accuracy"],
         "iteracion_2": v2["welfake_holdout"]["forzado"]["accuracy"]},
        {"vista": "Final · WELFake holdout F1 macro (forzado)", "iteracion_1": v1["welfake_holdout"]["forzado"]["macro_f1"],
         "iteracion_2": v2["welfake_holdout"]["forzado"]["macro_f1"]},
        {"vista": "Final · WELFake holdout trivial accuracy", "iteracion_1": v1["welfake_holdout"]["trivial"]["accuracy"],
         "iteracion_2": v2["welfake_holdout"]["trivial"]["accuracy"]},
        {"vista": "Final · WELFake holdout ECE (it2 contraída)", "iteracion_1": v1["welfake_holdout"]["forzado"]["ece"],
         "iteracion_2": v2["welfake_holdout"]["contraido"]["ece"]},
        {"vista": "Final · WELFake holdout cobertura del sistema", "iteracion_1": v1["welfake_holdout"]["sistema"]["cobertura"],
         "iteracion_2": v2["welfake_holdout"]["sistema"]["cobertura"]},
        {"vista": "Final · WELFake holdout accuracy respondidos", "iteracion_1": v1["welfake_holdout"]["sistema"]["accuracy_respondidos"],
         "iteracion_2": v2["welfake_holdout"]["sistema"]["accuracy_respondidos"]},
        {"vista": "Final · WELFake holdout trivial sobre respondidos", "iteracion_1": v1["welfake_holdout"]["sistema"]["trivial_respondidos"],
         "iteracion_2": v2["welfake_holdout"]["sistema"]["trivial_respondidos"]},
        {"vista": "Final · LIAR test accuracy (NO virgen)", "iteracion_1": v1["liar_test_no_virgen"]["forzado"]["accuracy"],
         "iteracion_2": v2["liar_test_no_virgen"]["forzado"]["accuracy"]},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(paths.reports / "comparison_it1_vs_it2.csv", index=False)
    return df


def success_criteria_v2() -> dict[str, Any]:
    """Criterios del acta reevaluados con el modelo final v2."""
    paths = get_paths()
    res = read_json(paths.reports / "final_v2.json")
    m = res["v2"]["isot_test"]["forzado"]
    triv = res["v2"]["isot_test"]["trivial"]
    it1 = res["it1"]["isot_test_iteracion1"]
    errs = pd.read_csv(paths.reports / "error_analysis_v2.csv", keep_default_na=False)
    reviewed = int((errs["comentario_equipo"].astype(str).str.strip() != "").sum())
    rows = [
        {"criterio": "Accuracy en test > 80 %", "valor": m["accuracy"], "estado": "cumplido" if m["accuracy"] > .8 else "no cumplido"},
        {"criterio": "F1 por clase > 0,80", "valor": [m["f1_real"], m["f1_fake"]],
         "estado": "cumplido" if min(m["f1_real"], m["f1_fake"]) > .8 else "no cumplido"},
        {"criterio": "Supera con claridad al ingenuo sin favorecer una clase", "valor": {"trivial_accuracy": triv["accuracy"]},
         "estado": "cumplido" if m["accuracy"] > triv["accuracy"] + .1 and min(m["f1_real"], m["f1_fake"]) > .8 else "no cumplido"},
        {"criterio": "Modelo adicional >= baseline, con la diferencia explicada",
         "valor": "iteración 1: SVM 0,988 vs LR 0,985 (McNemar p=0,030); el modelo v2 se elige por robustez, no por in-domain",
         "estado": "cumplido (iteración 1; ver informe §15 para el v2)"},
        {"criterio": "Sin filtración entre splits", "valor": "0 solapamientos ISOT; WELFake deduplicado contra todo ISOT y LIAR",
         "estado": "cumplido"},
        {"criterio": ">= 20 errores revisados por el equipo", "valor": {"casos_v2": len(errs), "comentados": reviewed},
         "estado": "cumplido" if reviewed >= 20 else "pendiente: revisión humana del equipo (error_analysis_v2.csv)"},
        {"criterio": "Demo devuelve etiqueta y probabilidad", "valor": "tests de predict y demo", "estado": "cumplido (si pytest pasa)"},
        {"criterio": "Accuracy > 98 % => sospecha", "valor": m["accuracy"],
         "estado": "ALERTA activa" if res["alerta_accuracy_sospechosa"]["alerta"] else "sin alerta (accuracy <= 98 %)"},
        {"criterio": "(it2) referencia it1 en ISOT test", "valor": it1["accuracy"], "estado": "informativo"},
    ]
    out = {"modelo": res["modelo_v2"], "criterios": rows, "generado": dt.datetime.now().isoformat(timespec="seconds")}
    write_json(out, paths.reports / "success_criteria_v2.json")
    return out
