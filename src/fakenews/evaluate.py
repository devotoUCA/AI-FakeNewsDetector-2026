"""Evaluación final en TEST (una vez por modelo, con registro de acceso), validación externa
en LIAR, análisis de errores y chequeo de los criterios de éxito del acta."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import plots
from .cleaning import clean_text, word_count
from .config import config_hash, get_paths, load_config, read_json, sha256_file, write_json
from .data import binarize_liar, load_liar
from .explain import global_top_terms, term_contributions
from .features import STYLE_FEATURES, StyleFeatures, english_stopword_ratio, known_vocab_ratio
from .metrics import bootstrap_ci, classification_metrics, compact, mcnemar_exact, reliability_bins
from .models import p_fake
from .splits import load_split
from .thresholds import accuracy_by_level, coverage_curve, zone_summary

LOGGER = logging.getLogger(__name__)

COMPARISON = ["dummy", "logreg_raw", "logreg_cal", "naive_bayes_cal", "linear_svm_cal", "style_tree",
              "style_forest", "ensemble", "knn", "mlp"]
PRETTY = {"dummy": "Trivial (mayoritaria)", "logreg_raw": "LR TF-IDF (baseline)",
          "logreg_cal": "LR TF-IDF calibrada", "naive_bayes_cal": "Naive Bayes", "linear_svm_cal": "SVM lineal",
          "style_tree": "Árbol (estilo)", "style_forest": "Bosque (estilo)", "ensemble": "Ensamble",
          "knn": "kNN coseno", "mlp": "Red pequeña (MLP)"}
# Tokens que delatan la fuente o la plataforma: no deberían estar entre los más influyentes.
SOURCE_TOKENS = {"reuters", "getty", "images", "featured", "21wire", "wire", "pic", "twitter com",
                 "http", "https", "www", "com", "acr", "boiler", "flickr", "via", "image",
                 "screenshot", "screen capture", "sap", "edited the statements"}


def _log_test_access(entries: list[dict[str, Any]], reason: str) -> None:
    """Registra cada acceso al test en `reports/test_access_log.jsonl` (append-only)."""
    paths = get_paths()
    log_path = paths.reports / "test_access_log.jsonl"
    previous = []
    if log_path.exists():
        previous = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    seen = {(e["modelo"], e["huella"]) for e in previous}
    with log_path.open("a", encoding="utf-8") as fh:
        for e in entries:
            e["reacceso"] = (e["modelo"], e["huella"]) in seen
            e["motivo"] = reason
            if e["reacceso"]:
                LOGGER.warning("RE-ACCESO al test para %s (misma huella). No usar para decidir.", e["modelo"])
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


def evaluate_final(reason: str = "evaluación final") -> dict[str, Any]:
    """Evalúa en test cada modelo final (una sola vez por huella de modelo) y arma los reportes."""
    cfg, paths = load_config(), get_paths()
    seed = cfg["seed"]
    test = load_split("test")
    y = test["label"].to_numpy()
    final = joblib.load(paths.models / "final_model.joblib")
    val_res = read_json(paths.reports / "val_results.json")
    manifest = read_json(paths.reports / "splits_manifest.json")
    ts = dt.datetime.now().isoformat(timespec="seconds")

    preds: dict[str, np.ndarray] = {}
    access = []
    for name in COMPARISON + ["logreg_minimal_leak"]:
        path = paths.models / "candidates" / f"{name}.joblib"
        art = joblib.load(path)
        preds[name] = p_fake(art["model"], test[art["input_col"]].tolist())
        access.append({"ts": ts, "modelo": name, "huella": sha256_file(path)[:16],
                       "config_hash": config_hash(), "test_sha256_ids": manifest["splits"]["test"]["sha256_ids"]})
    _log_test_access(access, reason)

    t = final["threshold"]
    lo, hi = final["ambiguous"]
    productive = final["name"]
    results: dict[str, Any] = {"productivo": productive, "umbral": t, "modelos": {}}
    for name in COMPARISON:
        p = preds[name]
        thr = t if name == productive else 0.5
        pred = (p >= thr).astype(int) if name != "knn" else (p > 0.5).astype(int)
        m = classification_metrics(y, pred, p)
        m["ic95"] = bootstrap_ci(y, pred, p, cfg["evaluation"]["bootstrap_resamples"], seed)
        m["umbral_usado"] = thr
        results["modelos"][name] = m
    base_pred = (preds["logreg_raw"] >= 0.5).astype(int)
    prod_pred = (preds[productive] >= t).astype(int)
    results["mcnemar"] = {
        f"logreg_raw (A) vs {productive} (B)": mcnemar_exact(y, base_pred, prod_pred),
        "logreg_raw (A) vs ensemble (B)": mcnemar_exact(y, base_pred, (preds["ensemble"] >= 0.5).astype(int)),
        "logreg_raw (A) vs mlp (B)": mcnemar_exact(y, base_pred, (preds["mlp"] >= 0.5).astype(int)),
        "logreg_raw (A) vs style_forest (B)": mcnemar_exact(y, base_pred, (preds["style_forest"] >= 0.5).astype(int)),
    }
    # Regla del acta para la red: adoptarla solo si mejora de forma consistente en val Y test.
    gain = cfg["models"]["mlp"]["adopt_min_gain_f1"]
    val_gain = val_res["mlp"]["fake"]["f1"] - val_res["logreg_raw"]["fake"]["f1"]
    test_gain = results["modelos"]["mlp"]["fake"]["f1"] - results["modelos"]["logreg_raw"]["fake"]["f1"]
    results["decision_red"] = {"ganancia_f1_val": val_gain, "ganancia_f1_test": test_gain,
                               "umbral_ganancia": gain,
                               "adoptar": bool(val_gain >= gain and test_gain >= gain)}

    # Sistema completo: zona ambigua, cobertura y niveles de confianza en test
    p = preds[productive]
    results["sistema_test"] = {"zona": zone_summary(y, p, t, lo, hi),
                               "cobertura": coverage_curve(y, p, t),
                               "por_nivel_confianza": accuracy_by_level(y, p, t, final["confidence_cuts"]),
                               "confiabilidad": reliability_bins(y, p, cfg["calibration"]["n_bins"])}

    # Experimento controlado de fuga en test (formulario 12: parámetro observado en prueba)
    pm = preds["logreg_minimal_leak"]
    results["experimento_fuga_test"] = {
        "antes (limpieza minimal, con marcas)": compact(classification_metrics(y, (pm >= 0.5).astype(int), pm)),
        "despues (limpieza full, sin marcas)": compact(results["modelos"]["logreg_raw"]),
    }

    # Alerta de accuracy sospechosa (acta: > 98 % se trata con sospecha) + diagnósticos
    acc = results["modelos"][productive]["accuracy"]
    results["alerta_accuracy_sospechosa"] = suspicious_accuracy_diagnostics(acc, final, test, results)

    # Figuras
    plots.confusion_matrix_plot(results["modelos"][productive]["confusion"],
                                f"Matriz de confusión en test — {PRETTY[productive]}",
                                paths.figures / "confusion_test_productive.png")
    plots.confusion_matrix_plot(results["modelos"]["logreg_raw"]["confusion"],
                                "Matriz de confusión en test — baseline LR",
                                paths.figures / "confusion_test_baseline.png")
    labels = [PRETTY[n] for n in COMPARISON]
    plots.bar_compare(labels, {"F1 falsa": [results["modelos"][n]["fake"]["f1"] for n in COMPARISON],
                               "F1 real": [results["modelos"][n]["real"]["f1"] for n in COMPARISON]},
                      "F1 por clase en test", "F1", paths.figures / "f1_by_model_test.png", (0, 1.02),
                      colors=[plots.FAKE, plots.REAL])
    plots.reliability_plot({"Test": results["sistema_test"]["confiabilidad"]},
                           f"Confiabilidad en test — {PRETTY[productive]}",
                           paths.figures / "reliability_productive_test.png")
    plots.coverage_plot(results["sistema_test"]["cobertura"], paths.figures / "coverage_test.png",
                        "Cobertura vs. accuracy en test")
    leak_val = read_json(paths.reports / "leakage_experiment_val.json")
    plots.bar_compare(["Validación", "Test"],
                      {"Con marcas (minimal)": [leak_val["val_minimal"]["f1_fake"],
                                                results["experimento_fuga_test"]["antes (limpieza minimal, con marcas)"]["f1_fake"]],
                       "Sin marcas (full)": [leak_val["val_full"]["f1_fake"],
                                             results["experimento_fuga_test"]["despues (limpieza full, sin marcas)"]["f1_fake"]]},
                      "Experimento de fuga: F1 de falsa", "F1 falsa",
                      paths.figures / "leakage_experiment.png", (0.9, 1.0))

    table = []
    for n in COMPARISON:
        m = results["modelos"][n]
        table.append({"modelo": n, **{k: round(v, 4) for k, v in compact(m).items()},
                      "ic95_f1_fake": [round(x, 4) for x in m["ic95"]["f1_fake"]],
                      "ic95_accuracy": [round(x, 4) for x in m["ic95"]["accuracy"]]})
    pd.DataFrame(table).to_csv(paths.reports / "test_comparison.csv", index=False)
    results["tabla_test"] = table
    write_json(results, paths.reports / "test_results.json")
    np.savez_compressed(paths.interim / "test_predictions.npz", y=y, **preds)

    errors = error_analysis(test, preds[productive], t, final)
    results["errores"] = errors
    return results


def suspicious_accuracy_diagnostics(acc: float, final: dict[str, Any], test: pd.DataFrame,
                                    results: dict[str, Any]) -> dict[str, Any]:
    """Si accuracy > umbral sospechoso, imprime alerta y corre diagnósticos de fuga."""
    limit = load_config()["evaluation"]["suspicious_accuracy"]
    out: dict[str, Any] = {"accuracy": acc, "umbral_sospecha": limit, "alerta": acc > limit}
    top = global_top_terms(final["model"], 50)
    terms = [d["termino"] for d in top["hacia_falsa"] + top["hacia_real"]]
    out["tokens_de_fuente_en_top50"] = sorted(set(terms) & SOURCE_TOKENS)
    from .audit import residual_marker_rates

    out["marcas_residuales_en_test"] = {k: v for k, v in residual_marker_rates(test["doc"]).items() if v > 0}
    exp = results["experimento_fuga_test"]
    out["caida_f1_al_quitar_marcas"] = (exp["antes (limpieza minimal, con marcas)"]["f1_fake"]
                                        - exp["despues (limpieza full, sin marcas)"]["f1_fake"])
    if out["alerta"]:
        LOGGER.warning("ALERTA: accuracy en test = %.4f > %.2f. Según el acta esto se trata con "
                       "sospecha: puede indicar que el modelo aprendió rasgos de la fuente o del "
                       "formato. Diagnósticos: tokens de fuente en top-50 = %s; marcas residuales "
                       "= %s; caída de F1 al quitar marcas = %.4f. Ver validación externa (LIAR).",
                       acc, limit, out["tokens_de_fuente_en_top50"], out["marcas_residuales_en_test"],
                       out["caida_f1_al_quitar_marcas"])
    return out


def _error_category(text: str, n_words: int, style: dict[str, float], y: int) -> str:
    """Categoría TENTATIVA del error (heurística; la revisión real es del equipo)."""
    low = text.lower()
    if n_words < 80:
        return "texto corto"
    if any(w in low for w in ("satire", "satirical", "parody", "the onion")):
        return "posible sátira"
    if style["attribution_ratio"] > 0.02 and y == 1:
        return "falsa con formato de agencia (muchas atribuciones)"
    if (style["exclam_per_100w"] > 0.3 or style["allcaps_word_ratio"] > 0.03) and y == 0:
        return "real con tono sensacionalista / mayúsculas"
    if style["first_person_ratio"] > 0.02:
        return "opinión / primera persona"
    if any(w in low for w in ("tweet", "twitter", "posted")):
        return "basada en redes sociales"
    return "tema o vocabulario ambiguo"


def error_analysis(test: pd.DataFrame, p: np.ndarray, t: float, final: dict[str, Any]) -> dict[str, Any]:
    """Genera `reports/error_analysis.csv` (>= N casos, FP y FN balanceados, con alta confianza)."""
    cfg, paths = load_config(), get_paths()
    n_min = cfg["evaluation"]["error_analysis_min_cases"]
    y = test["label"].to_numpy()
    pred = (p >= t).astype(int)
    df = test.assign(p_false=p, pred=pred, origen="test")
    errs = df[df["pred"] != df["label"]].copy()
    if len(errs) < n_min:  # se completa con errores de validación (marcados con su origen)
        val = load_split("val")
        art = joblib.load(paths.models / "final_model.joblib")
        pv = p_fake(art["model"], val["doc"].tolist())
        v = val.assign(p_false=pv, pred=(pv >= t).astype(int), origen="validacion")
        errs = pd.concat([errs, v[v["pred"] != v["label"]]], ignore_index=True)
    errs["tipo"] = np.where(errs["label"] == 1, "FN (falsa dada por real)", "FP (real marcada como falsa)")
    errs["confianza"] = np.maximum(errs["p_false"], 1 - errs["p_false"])
    fp = errs[errs["label"] == 0].sort_values("confianza", ascending=False)
    fn = errs[errs["label"] == 1].sort_values("confianza", ascending=False)
    half = max(n_min // 2, 1)
    chosen = pd.concat([fp.head(max(half, n_min - min(len(fn), half))),
                        fn.head(max(half, n_min - min(len(fp), half)))]).drop_duplicates("id")
    rows = []
    for _, r in chosen.iterrows():
        style = dict(zip(STYLE_FEATURES, StyleFeatures().transform([r["doc"]])[0]))
        terms = term_contributions(final["model"], r["doc"], top_k=5)
        cat = _error_category(r["doc"], int(r["n_words"]), style, int(r["label"]))
        rows.append({
            "id": r["id"], "origen": r["origen"], "tipo": r["tipo"],
            "etiqueta_real": "falsa" if r["label"] == 1 else "real",
            "prediccion": "falsa" if r["pred"] == 1 else "real", "p_false": round(float(r["p_false"]), 4),
            "confianza": round(float(r["confianza"]), 4), "alta_confianza": bool(r["confianza"] >= 0.8),
            "palabras": int(r["n_words"]), "subject_original": r["subject"],
            "titulo": r["doc"].split("\n")[0][:160],
            "texto_resumido": " ".join(r["doc"].split("\n")[1:])[:400],
            "terminos_hacia_falsa": "; ".join(d["termino"] for d in terms["hacia_falsa"]),
            "terminos_hacia_real": "; ".join(d["termino"] for d in terms["hacia_real"]),
            "mayusculas_pal": round(style["allcaps_word_ratio"], 4),
            "exclamaciones_100p": round(style["exclam_per_100w"], 3),
            "atribuciones": round(style["attribution_ratio"], 4),
            "categoria_tentativa": cat,
            "comentario_borrador": f"BORRADOR – revisar por el equipo. {_draft_comment(r, cat, terms)}",
            "comentario_equipo": "",
            "revisado_por": "",
        })
    out = pd.DataFrame(rows)
    out.to_csv(paths.reports / "error_analysis.csv", index=False)
    summary = {"n_errores_test": int((pred != y).sum()), "fp_test": int(((pred == 1) & (y == 0)).sum()),
               "fn_test": int(((pred == 0) & (y == 1)).sum()), "n_en_csv": len(out),
               "por_origen": out["origen"].value_counts().to_dict(),
               "por_tipo": out["tipo"].value_counts().to_dict(),
               "alta_confianza": int(out["alta_confianza"].sum()),
               "por_categoria": out["categoria_tentativa"].value_counts().to_dict()}
    write_json(summary, paths.reports / "error_analysis_summary.json")
    return summary


def _draft_comment(r: pd.Series, cat: str, terms: dict[str, list[dict[str, float]]]) -> str:
    push = terms["hacia_falsa"] if r["pred"] == 1 else terms["hacia_real"]
    words = ", ".join(d["termino"] for d in push[:3]) or "sin términos destacados"
    return (f"El modelo predijo {'falsa' if r['pred'] == 1 else 'real'} con p={r['p_false']:.2f}; "
            f"los términos que más empujaron fueron: {words}. Categoría tentativa: {cat}.")


def evaluate_external() -> dict[str, Any]:
    """Validación externa en LIAR, SIN reentrenar ni ajustar nada con ese dataset."""
    cfg, paths = load_config(), get_paths()
    liar_all = load_liar()
    liar = binarize_liar(liar_all)
    final = joblib.load(paths.models / "final_model.joblib")
    texts = [clean_text(s) for s in liar["statement"]]
    y = liar["y"].to_numpy()
    t, (lo, hi) = final["threshold"], final["ambiguous"]
    out: dict[str, Any] = {
        "binarizacion": {"falsa (1)": cfg["external"]["liar_fake_labels"],
                         "real (0)": cfg["external"]["liar_real_labels"],
                         "excluidas": cfg["external"]["liar_excluded_labels"]},
        "conteos_originales": liar_all["label"].value_counts().to_dict(),
        "n_evaluadas": int(len(liar)), "proporcion_falsas": float(y.mean()), "modelos": {}}
    for name in ["dummy", "logreg_raw", final["name"], "style_forest", "knn", "mlp"]:
        art = joblib.load(paths.models / "candidates" / f"{name}.joblib")
        p = p_fake(art["model"], texts)
        thr = t if name == final["name"] else 0.5
        m = classification_metrics(y, (p >= thr).astype(int), p)
        m["ic95"] = bootstrap_ci(y, (p >= thr).astype(int), p, 500, cfg["seed"])
        m["p_media"] = float(p.mean())
        out["modelos"][name] = m
        if name == final["name"]:
            p_prod = p
    out["sistema_con_zona"] = zone_summary(y, p_prod, t, lo, hi)
    n_words = np.array([word_count(x) for x in texts])
    kv = np.array([known_vocab_ratio(x, final["vocabulary"]) for x in texts])
    sw = np.array([english_stopword_ratio(x) for x in texts])
    ood = cfg["ood"]
    flagged = (n_words < ood["min_words"]) | (kv < ood["min_known_vocab_ratio"]) | \
              (sw < ood["min_english_stopword_ratio"])
    out["deriva_y_fuera_de_distribucion"] = {
        "palabras_mediana_liar": float(np.median(n_words)),
        "palabras_mediana_train": final["train_words_percentiles"]["50"],
        "frac_marcadas_como_texto_corto": float((n_words < ood["min_words"]).mean()),
        "frac_vocab_conocido_bajo": float((kv < ood["min_known_vocab_ratio"]).mean()),
        "frac_con_alguna_advertencia_que_fuerza_incierta": float(flagged.mean()),
        "vocab_conocido_mediana": float(np.median(kv)),
        "nota": "En la demo, estas entradas se responderían como 'incierta' por estar fuera de "
                "distribución; las métricas de arriba fuerzan igual una decisión para medir.",
    }
    m_long = n_words >= 20
    if m_long.sum() >= 30:
        out["solo_afirmaciones_de_20_o_mas_palabras"] = {
            "n": int(m_long.sum()),
            **compact(classification_metrics(y[m_long], (p_prod[m_long] >= t).astype(int), p_prod[m_long]))}
    write_json(out, paths.reports / "external_liar.json")
    labels = [PRETTY.get(k, k) for k in out["modelos"]]
    plots.bar_compare(labels, {"Accuracy": [v["accuracy"] for v in out["modelos"].values()],
                               "F1 falsa": [v["fake"]["f1"] for v in out["modelos"].values()],
                               "F1 real": [v["real"]["f1"] for v in out["modelos"].values()]},
                      "Validación externa en LIAR (sin reentrenar)", "Valor",
                      paths.figures / "external_liar.png", (0, 1),
                      colors=[plots.NEUTRAL, plots.FAKE, plots.REAL])
    return out


def success_criteria() -> dict[str, Any]:
    """Chequea los criterios de éxito del acta contra los reportes reales (lab 22)."""
    cfg, paths = load_config(), get_paths()
    ev = cfg["evaluation"]
    test = read_json(paths.reports / "test_results.json")
    manifest = read_json(paths.reports / "splits_manifest.json")
    prod = test["productivo"]
    m = test["modelos"][prod]
    dummy = test["modelos"]["dummy"]
    errs = pd.read_csv(paths.reports / "error_analysis.csv", keep_default_na=False)
    reviewed = int((errs["comentario_equipo"].astype(str).str.strip() != "").sum())
    mc = test["mcnemar"][f"logreg_raw (A) vs {prod} (B)"]
    rows = [
        {"criterio": "Accuracy en test > 80 %", "valor": m["accuracy"],
         "estado": "cumplido" if m["accuracy"] > ev["success_accuracy"] else "no cumplido"},
        {"criterio": "F1 por clase > 0,80 (real y falsa)", "valor": [m["real"]["f1"], m["fake"]["f1"]],
         "estado": "cumplido" if min(m["real"]["f1"], m["fake"]["f1"]) > ev["success_f1_per_class"] else "no cumplido"},
        {"criterio": "Supera con claridad al clasificador ingenuo sin favorecer una clase",
         "valor": {"accuracy_trivial": dummy["accuracy"], "f1_falsa_trivial": dummy["fake"]["f1"]},
         "estado": "cumplido" if m["accuracy"] > dummy["accuracy"] + 0.1 and min(m["real"]["f1"], m["fake"]["f1"]) > 0.8 else "no cumplido"},
        {"criterio": "Modelo adicional iguala o supera al baseline, con la diferencia explicada",
         "valor": {"f1_falsa_baseline": test["modelos"]["logreg_raw"]["fake"]["f1"],
                   "f1_falsa_productivo": m["fake"]["f1"], "mcnemar_p": mc["p_value"]},
         "estado": "cumplido (ver informe para la explicación)" if m["fake"]["f1"] >= test["modelos"]["logreg_raw"]["fake"]["f1"] - 1e-9 else "no cumplido"},
        {"criterio": "Train / validación / test sin filtración", "valor": manifest["solapamiento"],
         "estado": "cumplido" if manifest["solapamiento"]["ok"] else "no cumplido"},
        {"criterio": ">= 20 errores revisados y comentados por el equipo",
         "valor": {"casos_en_csv": len(errs), "comentados_por_el_equipo": reviewed},
         "estado": "cumplido" if reviewed >= 20 else "pendiente: requiere revisión humana del equipo"},
        {"criterio": "Demo devuelve etiqueta y probabilidad para un texto nuevo",
         "valor": "verificado por tests/test_demo.py y tests/test_predict.py",
         "estado": "cumplido (si la suite de tests pasa)"},
        {"criterio": "Alerta de accuracy > 98 % (tratar con sospecha)",
         "valor": test["alerta_accuracy_sospechosa"]["accuracy"],
         "estado": "ALERTA activa: ver diagnósticos y validación externa" if test["alerta_accuracy_sospechosa"]["alerta"] else "sin alerta"},
    ]
    out = {"modelo_productivo": prod, "criterios": rows,
           "generado": dt.datetime.now().isoformat(timespec="seconds")}
    write_json(out, paths.reports / "success_criteria.json")
    return out


def text_fingerprint(text: str) -> str:
    """Hash corto de un texto (para registros sin guardar contenido)."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]
