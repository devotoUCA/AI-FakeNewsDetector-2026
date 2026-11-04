"""Entrenamiento y selección de modelos. TODAS las decisiones se toman en validación.

Salidas:
* `models/candidates/*.joblib`: cada modelo de comparación entrenado en train.
* `models/final_model.joblib` y `models/retrieval.joblib`: sistema productivo.
* `models/metadata.json`: trazabilidad (hashes de datos, semilla, versiones, decisiones).
* `reports/val_*.json` y figuras.
"""

from __future__ import annotations

import copy
import datetime as dt
import logging
import platform
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import models as M
from . import plots
from .config import config_hash, get_paths, load_config, read_json, set_seed, write_json
from .explain import global_top_terms
from .features import STYLE_FEATURES, StyleFeatures, make_tfidf
from .metrics import classification_metrics, compact, reliability_bins
from .scratch import ScratchLogisticRegression
from .splits import load_split
from .thresholds import (choose_confidence_cuts, choose_threshold, coverage_curve, pr_curve,
                         zone_summary)

LOGGER = logging.getLogger(__name__)


def _f1_fake(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, pos_label=1, zero_division=0))


def _row(name: str, params: dict[str, Any], ytr: np.ndarray, ptr_pred: np.ndarray, yva: np.ndarray,
         pva_pred: np.ndarray, pva: np.ndarray | None = None) -> dict[str, Any]:
    m_tr = classification_metrics(ytr, ptr_pred)
    m_va = classification_metrics(yva, pva_pred, pva)
    return {"modelo": name, **params, "f1_fake_train": m_tr["fake"]["f1"],
            "accuracy_train": m_tr["accuracy"], "f1_fake_val": m_va["fake"]["f1"],
            "accuracy_val": m_va["accuracy"], "precision_fake_val": m_va["fake"]["precision"],
            "recall_fake_val": m_va["fake"]["recall"], "f1_real_val": m_va["real"]["f1"],
            "brecha_f1_train_val": m_tr["fake"]["f1"] - m_va["fake"]["f1"]}


def _select(rows: list[dict[str, Any]], max_gap: float | None = None) -> dict[str, Any]:
    """Mejor F1 de falsa en validación; si hay tope de brecha train-val, se respeta."""
    pool = [r for r in rows if max_gap is None or r["brecha_f1_train_val"] <= max_gap] or rows
    return max(pool, key=lambda r: (round(r["f1_fake_val"], 6), -r.get("C", 0)))


def _save(model: Any, name: str, input_col: str = "doc") -> None:
    paths = get_paths()
    out = paths.models / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"name": name, "model": M.strip_heavy_attributes(model), "input_col": input_col},
                out / f"{name}.joblib", compress=3)


def train_all() -> dict[str, Any]:
    """Entrena todos los modelos, calibra, elige umbral y guarda artefactos y reportes."""
    cfg, paths = load_config(), get_paths()
    seed = set_seed()
    mc = cfg["models"]
    t0 = time.time()
    train, val = load_split("train"), load_split("val")
    Xtr, Xva = train["doc"].tolist(), val["doc"].tolist()
    ytr, yva = train["label"].to_numpy(), val["label"].to_numpy()
    LOGGER.info("Train: %d | Val: %d", len(Xtr), len(Xva))
    report: dict[str, Any] = {"config_hash": config_hash(), "seed": seed}
    val_preds: dict[str, np.ndarray] = {}

    # --- 0. Piso: clasificador trivial ----------------------------------------------------
    dmy = M.dummy().fit(Xtr, ytr)
    p = M.p_fake(dmy, Xva)
    report["dummy"] = classification_metrics(yva, dmy.predict(Xva), p)
    report["dummy"]["proporcion_falsas_train"] = float(ytr.mean())
    val_preds["dummy"] = p
    _save(dmy, "dummy")

    # --- 1. Baseline: TF-IDF + regresión logística (grilla C x n-gramas) -------------------
    lr_rows, nb_rows, svm_rows = [], [], []
    mats: dict[tuple[int, int], tuple[Any, Any]] = {}
    for ngram in map(tuple, mc["logreg"]["ngram_grid"]):
        LOGGER.info("TF-IDF ngram=%s ...", ngram)
        vec = make_tfidf(ngram)
        A, B = vec.fit_transform(Xtr), vec.transform(Xva)
        mats[ngram] = (A, B)
        for C in mc["logreg"]["C_grid"]:
            clf = M.logreg_clf(C).fit(A, ytr)
            lr_rows.append(_row("logreg", {"C": C, "ngram": list(ngram), "n_features": A.shape[1]},
                                ytr, clf.predict(A), yva, clf.predict(B), clf.predict_proba(B)[:, 1]))
        for alpha in mc["naive_bayes"]["alpha_grid"]:
            clf = M.naive_bayes(alpha, ngram).steps[-1][1].fit(A, ytr)
            nb_rows.append(_row("naive_bayes", {"alpha": alpha, "ngram": list(ngram)}, ytr,
                                clf.predict(A), yva, clf.predict(B), clf.predict_proba(B)[:, 1]))
        for C in mc["linear_svm"]["C_grid"]:
            clf = M.linear_svm_clf(C).fit(A, ytr)
            svm_rows.append(_row("linear_svm", {"C": C, "ngram": list(ngram)}, ytr, clf.predict(A),
                                 yva, clf.predict(B)))
    best_lr = _select(lr_rows, mc["logreg"]["max_train_val_gap"])
    best_nb, best_svm = _select(nb_rows), _select(svm_rows)
    report["grillas"] = {"logreg": lr_rows, "naive_bayes": nb_rows, "linear_svm": svm_rows}
    report["seleccion"] = {"logreg": best_lr, "naive_bayes": best_nb, "linear_svm": best_svm}
    pd.DataFrame(lr_rows).to_csv(paths.reports / "baseline_logreg_grid.csv", index=False)
    LOGGER.info("Mejor LR: C=%s ngram=%s F1 val=%.4f", best_lr["C"], best_lr["ngram"],
                best_lr["f1_fake_val"])

    lr_ngram = tuple(best_lr["ngram"])
    lr_raw = M.logreg(best_lr["C"], lr_ngram).fit(Xtr, ytr)
    p = M.p_fake(lr_raw, Xva)
    report["logreg_raw"] = classification_metrics(yva, (p >= 0.5).astype(int), p)
    val_preds["logreg_raw"] = p
    top_terms = global_top_terms(lr_raw, 30)
    write_json(top_terms, paths.reports / "top_terms_baseline.json")
    _save(lr_raw, "logreg_raw")

    # --- 2. Experimento controlado de fuga (mismo modelo, misma división; cambia la limpieza)
    LOGGER.info("Experimento controlado de fuga (limpieza minimal vs full)...")
    lr_min = M.logreg(best_lr["C"], lr_ngram).fit(train["doc_minimal"].tolist(), ytr)
    p_min = M.p_fake(lr_min, val["doc_minimal"].tolist())
    leak = {
        "descripcion": "Mismo modelo (LR, mismos C y n-gramas), mismos datos, misma semilla y misma "
                       "división. Cambia solo la limpieza: 'minimal' conserva marcas de fuente, de "
                       "formato y de plataforma; 'full' las quita.",
        "val_minimal": compact(classification_metrics(yva, (p_min >= 0.5).astype(int), p_min)),
        "val_full": compact(report["logreg_raw"]),
        "top_terms_minimal": global_top_terms(lr_min, 20),
    }
    write_json(leak, paths.reports / "leakage_experiment_val.json")
    _save(lr_min, "logreg_minimal_leak", input_col="doc_minimal")

    # --- 3. Rasgos de estilo: árbol y bosque -----------------------------------------------
    LOGGER.info("Rasgos de estilo...")
    S_tr, S_va = StyleFeatures().transform(Xtr), StyleFeatures().transform(Xva)
    tree_rows, forest_rows = [], []
    for d in mc["style_tree"]["max_depth_grid"]:
        clf = M.style_tree_clf(d).fit(S_tr, ytr)
        tree_rows.append(_row("style_tree", {"max_depth": d}, ytr, clf.predict(S_tr), yva,
                              clf.predict(S_va), clf.predict_proba(S_va)[:, 1]))
    for d in mc["style_forest"]["max_depth_grid"]:
        clf = M.style_forest_clf(d).fit(S_tr, ytr)
        forest_rows.append(_row("style_forest", {"max_depth": d}, ytr, clf.predict(S_tr), yva,
                                clf.predict(S_va), clf.predict_proba(S_va)[:, 1]))
    best_tree, best_forest = _select(tree_rows), _select(forest_rows)
    report["grillas"].update({"style_tree": tree_rows, "style_forest": forest_rows})
    report["seleccion"].update({"style_tree": best_tree, "style_forest": best_forest})
    tree = M.style_tree(best_tree["max_depth"]).fit(Xtr, ytr)
    forest = M.style_forest(best_forest["max_depth"]).fit(Xtr, ytr)
    for name, mdl in (("style_tree", tree), ("style_forest", forest)):
        p = M.p_fake(mdl, Xva)
        report[name] = classification_metrics(yva, (p >= 0.5).astype(int), p)
        val_preds[name] = p
        _save(mdl, name)
    report["style_importancias"] = dict(sorted(zip(
        STYLE_FEATURES, forest.named_steps["clf"].feature_importances_.round(4).tolist()),
        key=lambda kv: -kv[1]))
    report["style_medias_por_clase_train"] = {
        "falsa": dict(zip(STYLE_FEATURES, S_tr[ytr == 1].mean(0).round(4).tolist())),
        "real": dict(zip(STYLE_FEATURES, S_tr[ytr == 0].mean(0).round(4).tolist()))}
    # Lab 09: kNN sobre rasgos sin estandarizar vs estandarizados (efecto de la escala)
    scale_rows = {}
    for label, pipe in (("sin_estandarizar", KNeighborsClassifier(15)),
                        ("estandarizado", Pipeline([("sc", StandardScaler()),
                                                    ("knn", KNeighborsClassifier(15))]))):
        pipe.fit(S_tr, ytr)
        scale_rows[label] = compact(classification_metrics(yva, pipe.predict(S_va)))
    report["knn_estilo_escala"] = scale_rows

    # --- 4. Ensamble (stacking texto + estilo) y análisis de desacuerdo ---------------------
    LOGGER.info("Ensamble (stacking LR texto + bosque de estilo)...")
    style_best = forest if best_forest["f1_fake_val"] >= best_tree["f1_fake_val"] else tree
    style_name = "style_forest" if style_best is forest else "style_tree"
    ens = M.stacking(M.logreg(best_lr["C"], lr_ngram), copy.deepcopy(style_best)).fit(Xtr, ytr)
    p = M.p_fake(ens, Xva)
    report["ensemble"] = classification_metrics(yva, (p >= 0.5).astype(int), p)
    report["ensemble"]["componentes"] = ["logreg (texto)", style_name]
    report["ensemble"]["pesos_meta"] = ens.final_estimator_.coef_.round(4).tolist()
    val_preds["ensemble"] = p
    _save(ens, "ensemble")
    report["desacuerdo"] = disagreement_analysis(val, val_preds["logreg_raw"], val_preds[style_name],
                                                 p, S_va)

    # --- 5. kNN con distancia coseno (k elegido en validación) ------------------------------
    LOGGER.info("kNN coseno...")
    knn_vec = make_tfidf((1, 1), max_features=mc["knn"]["max_features"])
    K_tr, K_va = knn_vec.fit_transform(Xtr), knn_vec.transform(Xva)
    kmax = max(mc["knn"]["k_grid"])
    nn = M.knn_clf(kmax).fit(K_tr, ytr)
    _, idx = nn.kneighbors(K_va, n_neighbors=kmax)
    neigh_labels = ytr[idx]
    knn_rows = []
    for k in mc["knn"]["k_grid"]:
        pk = neigh_labels[:, :k].mean(axis=1)
        knn_rows.append({"modelo": "knn", "k": k, **{f"{kk}_val": v for kk, v in
                         compact(classification_metrics(yva, (pk > 0.5).astype(int), pk)).items()}})
    best_k = max(knn_rows, key=lambda r: (round(r["f1_fake_val"], 6), -r["k"]))
    report["grillas"]["knn"] = knn_rows
    report["seleccion"]["knn"] = best_k
    same_topic_diff_label = float((neigh_labels[:, :best_k["k"]] != yva[:, None]).mean())
    knn = M.knn(best_k["k"]).fit(Xtr, ytr)
    p = M.p_fake(knn, Xva)
    report["knn"] = classification_metrics(yva, (p > 0.5).astype(int), p)
    report["knn"]["frac_vecinos_con_otra_etiqueta"] = same_topic_diff_label
    val_preds["knn"] = p
    _save(knn, "knn")
    retrieval = {"pipeline": M.strip_heavy_attributes(knn), "ids": train["id"].tolist(),
                 "titles": train["doc"].str.split("\n").str[0].str[:160].tolist(),
                 "labels": ytr.tolist()}

    # --- 6. Red neuronal pequeña (comparación) con curvas por época -------------------------
    LOGGER.info("Red neuronal pequeña (MLP)...")
    mlp_res = train_mlp(Xtr, ytr, Xva, yva)
    report["mlp"] = mlp_res["metricas_val"]
    report["mlp_curvas"] = mlp_res["curvas"]
    val_preds["mlp"] = mlp_res["p_val"]
    _save(mlp_res["pipeline"], "mlp")

    # --- 7. Regresión logística desde cero (labs 07, 11-13) -------------------------------
    LOGGER.info("Regresión logística desde cero (gradiente)...")
    report["scratch"] = scratch_comparison(Xtr, ytr, Xva, yva)

    # --- 8. Calibración (CV dentro de train) y elección por Brier en validación -------------
    LOGGER.info("Calibración...")
    cal_report: dict[str, Any] = {}
    calibrated_models: dict[str, Any] = {}
    builders = {"logreg": lambda: M.logreg(best_lr["C"], lr_ngram),
                "naive_bayes": lambda: M.naive_bayes(best_nb["alpha"], tuple(best_nb["ngram"])),
                "linear_svm": lambda: M.linear_svm(best_svm["C"], tuple(best_svm["ngram"]))}
    for name, build in builders.items():
        options: dict[str, Any] = {}
        for method in cfg["calibration"]["methods"]:
            if method == "none":
                if name == "linear_svm":
                    continue  # el SVM no tiene probabilidad sin calibrar
                mdl = lr_raw if name == "logreg" else build().fit(Xtr, ytr)
            else:
                mdl = M.calibrated(build(), method).fit(Xtr, ytr)
            p = M.p_fake(mdl, Xva)
            options[method] = {"modelo": mdl, "p": p,
                               "metricas": classification_metrics(yva, (p >= 0.5).astype(int), p),
                               "confiabilidad": reliability_bins(yva, p, cfg["calibration"]["n_bins"])}
        chosen = min(options, key=lambda k: options[k]["metricas"]["brier"])
        cal_report[name] = {"elegido": chosen, "criterio": "menor Brier en validación",
                            **{k: {"brier": v["metricas"]["brier"], "log_loss": v["metricas"]["log_loss"],
                                   "ece": v["metricas"]["ece"], "f1_fake": v["metricas"]["fake"]["f1"],
                                   "confiabilidad": v["confiabilidad"]} for k, v in options.items()}}
        calibrated_models[name] = options[chosen]["modelo"]
        val_preds[f"{name}_cal"] = options[chosen]["p"]
        report[f"{name}_cal"] = options[chosen]["metricas"]
        report[f"{name}_cal"]["calibracion"] = chosen
        _save(options[chosen]["modelo"], f"{name}_cal")
        if name == "logreg":
            curves = {"Sin calibrar": options["none"]["confiabilidad"]}
            if chosen != "none":
                curves[f"Calibrado ({chosen})"] = options[chosen]["confiabilidad"]
            plots.reliability_plot(curves, "Confiabilidad en validación (regresión logística)",
                                   paths.figures / "reliability_logreg_val.png")
    write_json(cal_report, paths.reports / "calibration_val.json")

    # --- 9. Modelo productivo: mejor F1 falsa en validación entre los candidatos calibrados -
    candidates = ["logreg_cal", "naive_bayes_cal", "linear_svm_cal", "ensemble"]
    scores = {c: report[c]["fake"]["f1"] for c in candidates}
    best_score = max(scores.values())
    # Empates (< 0,002 de F1) se resuelven a favor del modelo más simple (orden de la lista).
    productive = next(c for c in candidates if scores[c] >= best_score - 0.002)
    prod_model = {"logreg_cal": calibrated_models["logreg"],
                  "naive_bayes_cal": calibrated_models["naive_bayes"],
                  "linear_svm_cal": calibrated_models["linear_svm"], "ensemble": ens}[productive]
    p_prod = val_preds[productive]
    report["productivo"] = {"modelo": productive, "f1_fake_val_por_candidato": scores,
                            "criterio": "mayor F1 de falsa en validación (umbral 0,5); empates "
                                        "(<0,002) -> el más simple"}

    # --- 10. Umbral, zona ambigua y niveles de confianza (validación) -----------------------
    dc = cfg["decision"]
    thr = choose_threshold(yva, p_prod, dc["threshold_grid"], dc["max_fpr"])
    t = thr["umbral"]
    conf = choose_confidence_cuts(yva, p_prod, t, dc["ambiguous_low"], dc["ambiguous_high"],
                                  dc["confidence_grid"], dc["confidence_high_target_accuracy"],
                                  window=dc["confidence_window"])
    zone = zone_summary(yva, p_prod, t, dc["ambiguous_low"], dc["ambiguous_high"])
    cov = coverage_curve(yva, p_prod, t)
    pr = pr_curve(yva, p_prod)
    write_json({"umbral": thr, "confianza": conf, "zona_ambigua": zone, "cobertura": cov},
               paths.reports / "decision_val.json")
    plots.pr_curve_plot(pr, thr["elegido"], paths.figures / "pr_curve_val.png")
    plots.coverage_plot(cov, paths.figures / "coverage_val.png",
                        "Cobertura vs. accuracy en validación (modelo productivo)")
    report["decision"] = {"umbral": t, "zona": [dc["ambiguous_low"], dc["ambiguous_high"]],
                          "cortes_confianza": conf["cortes"], "zona_resumen": zone}

    # --- 11. Artefactos productivos ----------------------------------------------------------
    vocab_vec = CountVectorizer(lowercase=True, min_df=cfg["tfidf"]["min_df"]).fit(Xtr)
    n_words = train["n_words"].to_numpy()
    meta = build_metadata(productive, t, conf["cortes"], report)
    final = {"name": productive, "model": M.strip_heavy_attributes(prod_model), "threshold": t,
             "ambiguous": [dc["ambiguous_low"], dc["ambiguous_high"]],
             "confidence_cuts": conf["cortes"], "vocabulary": frozenset(vocab_vec.vocabulary_),
             "train_words_percentiles": {str(q): float(np.percentile(n_words, q))
                                         for q in (1, 5, 50, 95, 99)},
             "metadata": meta}
    joblib.dump(final, paths.models / "final_model.joblib", compress=3)
    joblib.dump(retrieval, paths.models / "retrieval.joblib", compress=3)
    write_json(meta, paths.models / "metadata.json")

    # --- 12. Tabla comparativa en validación ------------------------------------------------
    order = ["dummy", "logreg_raw", "logreg_cal", "naive_bayes_cal", "linear_svm_cal", "style_tree",
             "style_forest", "ensemble", "knn", "mlp"]
    table = [{"modelo": k, **compact(report[k])} for k in order]
    pd.DataFrame(table).to_csv(paths.reports / "val_comparison.csv", index=False)
    report["tabla_val"] = table
    report["segundos"] = round(time.time() - t0, 1)
    write_json(report, paths.reports / "val_results.json")
    np.savez_compressed(paths.interim / "val_predictions.npz", y=yva, **val_preds)
    LOGGER.info("Entrenamiento completo en %.0f s. Productivo: %s (umbral %.2f)", time.time() - t0,
                productive, t)
    return report


def train_mlp(Xtr: list[str], ytr: np.ndarray, Xva: list[str], yva: np.ndarray) -> dict[str, Any]:
    """Entrena la red época por época (partial_fit) y conserva la mejor época según val."""
    c = load_config()["models"]["mlp"]
    vec = make_tfidf((1, 2), max_features=c["max_features"])
    A, B = vec.fit_transform(Xtr), vec.transform(Xva)
    mlp = M.mlp_clf()
    curves, best, best_loss = [], None, np.inf
    for epoch in range(1, c["epochs"] + 1):
        mlp.partial_fit(A, ytr, classes=[0, 1])
        ptr, pva = mlp.predict_proba(A)[:, 1], mlp.predict_proba(B)[:, 1]
        mtr = classification_metrics(ytr, (ptr >= 0.5).astype(int), ptr)
        mva = classification_metrics(yva, (pva >= 0.5).astype(int), pva)
        curves.append({"epoca": epoch, "loss_train": mtr["log_loss"], "loss_val": mva["log_loss"],
                       "f1_fake_train": mtr["fake"]["f1"], "f1_fake_val": mva["fake"]["f1"]})
        if mva["log_loss"] < best_loss:
            best_loss, best = mva["log_loss"], copy.deepcopy(mlp)
    pipe = Pipeline([("tfidf", vec), ("clf", best)])
    p = M.p_fake(pipe, Xva)
    paths = get_paths()
    plots.line_plot({"Entrenamiento": ([r["epoca"] for r in curves], [r["loss_train"] for r in curves]),
                     "Validación": ([r["epoca"] for r in curves], [r["loss_val"] for r in curves])},
                    "Red pequeña: log-loss por época", "Época", "Log-loss", paths.figures /
                    "mlp_learning_curve.png", logy=True)
    best_epoch = min(curves, key=lambda r: r["loss_val"])["epoca"]
    return {"pipeline": pipe, "p_val": p, "curvas": {"por_epoca": curves, "mejor_epoca": best_epoch},
            "metricas_val": classification_metrics(yva, (p >= 0.5).astype(int), p)}


def scratch_comparison(Xtr: list[str], ytr: np.ndarray, Xva: list[str], yva: np.ndarray
                       ) -> dict[str, Any]:
    """Compara la regresión logística desde cero (varias tasas de aprendizaje) con sklearn."""
    c = load_config()["models"]["scratch_logreg"]
    vec = make_tfidf((1, 1), max_features=c["max_features"])
    A, B = vec.fit_transform(Xtr), vec.transform(Xva)
    out: dict[str, Any] = {"tasas": {}}
    series = {}
    for lr in c["learning_rates"]:
        m = ScratchLogisticRegression(learning_rate=lr, epochs=c["epochs"]).fit(A, ytr, B, yva)
        pv = m.predict_proba(B)
        out["tasas"][str(lr)] = {"metricas_val": compact(classification_metrics(yva, (pv >= 0.5).astype(int), pv)),
                                 "historia": m.history}
        series[f"tasa {lr} (val)"] = ([h["epoca"] for h in m.history], [h["loss_val"] for h in m.history])
        last = m
    sk = M.logreg_clf(1.0).fit(A, ytr)
    out["sklearn_logreg_C1"] = compact(classification_metrics(yva, sk.predict(B), sk.predict_proba(B)[:, 1]))
    out["acuerdo_predicciones_mejor_tasa_vs_sklearn"] = float((last.predict(B) == sk.predict(B)).mean())
    plots.line_plot(series, "Regresión logística desde cero: pérdida por época", "Época",
                    "Log-loss (validación)", get_paths().figures / "scratch_learning_rates.png")
    write_json(out, get_paths().reports / "scratch_vs_sklearn.json")
    return {k: v for k, v in out.items() if k != "tasas"} | {
        "tasas_resumen": {k: v["metricas_val"] for k, v in out["tasas"].items()}}


def disagreement_analysis(val: pd.DataFrame, p_text: np.ndarray, p_style: np.ndarray,
                          p_ens: np.ndarray, S_va: np.ndarray) -> dict[str, Any]:  # noqa: N803
    """Casos donde el modelo de texto y el de estilo dan clases distintas (acta, formulario 8)."""
    y = val["label"].to_numpy()
    a, b, e = (p_text >= 0.5).astype(int), (p_style >= 0.5).astype(int), (p_ens >= 0.5).astype(int)
    dis = a != b
    df = pd.DataFrame({"id": val["id"], "etiqueta": y, "p_texto": p_text.round(4),
                       "p_estilo": p_style.round(4), "p_ensamble": p_ens.round(4),
                       "titulo": val["doc"].str.split("\n").str[0].str[:150]})
    for j, f in enumerate(STYLE_FEATURES):
        df[f] = S_va[:, j].round(4)
    sub = df[dis].copy()
    sub["acierta"] = np.where(a[dis] == y[dis], "texto", "estilo")
    sub.sort_values("p_ensamble").to_csv(get_paths().reports / "disagreement_val.csv", index=False)
    style_means = {g: sub.loc[sub["acierta"] == g, STYLE_FEATURES].mean().round(4).to_dict()
                   for g in ("texto", "estilo")}
    return {"n_desacuerdos": int(dis.sum()), "fraccion": float(dis.mean()),
            "acierta_texto": int((a[dis] == y[dis]).sum()), "acierta_estilo": int((b[dis] == y[dis]).sum()),
            "acierta_ensamble": int((e[dis] == y[dis]).sum()),
            "medias_estilo_segun_quien_acierta": style_means}


def build_metadata(productive: str, threshold: float, cuts: dict[str, float],
                   report: dict[str, Any]) -> dict[str, Any]:
    """Metadata de trazabilidad del modelo productivo (lab 21: versión del modelo)."""
    paths = get_paths()
    manifest = read_json(paths.reports / "splits_manifest.json")
    import scipy

    return {
        "modelo": productive, "version": f"{productive}-{config_hash()}",
        "creado": dt.datetime.now().isoformat(timespec="seconds"), "semilla": load_config()["seed"],
        "config_hash": config_hash(), "umbral": threshold, "cortes_confianza": cuts,
        "zona_ambigua": [load_config()["decision"]["ambiguous_low"],
                         load_config()["decision"]["ambiguous_high"]],
        "datos": {k: {"sha256_ids": v["sha256_ids"], "n": v["total"]} for k, v in manifest["splits"].items()},
        "metricas_val": compact(report[productive]),
        "versiones": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                      "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
    }
