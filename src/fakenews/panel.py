"""Panel de evaluación de la iteración 2 y selección robusta del modelo.

Vistas (ninguna usa el test de ISOT ni el holdout de WELFake):
1. in_domain: entrena en ISOT train (+LIAR train si es D1) y evalúa en ISOT val.
2. grupo: 4 folds que retienen un `subject` real y un grupo de `subject` falsos (pool = train+val).
3. tiempo: entrena con noticias anteriores al corte y evalúa con las posteriores (pool = train+val).
4. externo_dev: WELFake-dev (artículos de otras fuentes, deduplicados contra ISOT).
4b. liar_valid: LIAR valid (se reporta; no entra al puntaje porque D1 entrena con LIAR).
5. invarianza: tasa de cambio de predicción ante transformaciones que no cambian la veracidad.

`subject` y `date` se usan SOLO para armar folds; nunca entran al modelo.

Criterio: puntaje_robusto = promedio de F1 macro en vistas 2, 3 y 4; pisos duros: F1 por clase
in-domain >= 0,90 y tasa máxima de cambio por invarianza <= 2 %; empate (< 0,005) -> más simple.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from . import models as M
from .cleaning import build_document
from .config import get_paths, load_config, write_json
from .corpus import isot_raw, isot_split, liar_split, welfake_split
from .features import make_tfidf
from .invariance import TRANSFORMS, TRANSFORMS_V3, flip_stats, flip_stats_ci
from .metrics import classification_metrics

LOGGER = logging.getLogger(__name__)
COMPLEXITY = {"lr_word": 1, "lr_char": 2, "svm_word": 3, "ensemble": 4, "emb_lr": 5}


# --- Construcción de modelos ---------------------------------------------------------------
def build_model(spec: dict[str, Any]) -> Any:
    """Construye el estimador de un candidato del panel."""
    kind, seed = spec["model"], load_config()["seed"]
    if kind == "lr_word":
        return Pipeline([("tfidf", make_tfidf(tuple(spec.get("ngram", (1, 2))))), ("clf", M.logreg_clf(spec["C"]))])
    if kind == "lr_char":
        t = load_config()["tfidf"]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True,
                              min_df=5, max_df=t["max_df"], max_features=spec.get("max_features", 150000),
                              dtype=np.float32)
        return Pipeline([("tfidf", vec), ("clf", M.logreg_clf(spec["C"]))])
    if kind == "svm_word":
        base = Pipeline([("tfidf", make_tfidf((1, 2))), ("clf", M.linear_svm_clf(spec["C"]))])
        cv = StratifiedKFold(n_splits=spec.get("cal_folds", 5), shuffle=True, random_state=seed)
        return CalibratedClassifierCV(base, method="isotonic", cv=cv, n_jobs=1)
    if kind == "ensemble":
        text = Pipeline([("tfidf", make_tfidf((1, 2))), ("clf", M.logreg_clf(spec["C"]))])
        style = M.style_forest(spec.get("style_depth", 12))
        style.named_steps["clf"].set_params(n_jobs=1)
        return M.stacking(text, style)
    if kind == "emb_lr":
        from .embeddings import EmbeddingLookup

        return Pipeline([("emb", EmbeddingLookup()), ("clf", M.logreg_clf(spec["C"]))])
    raise ValueError(f"Modelo desconocido: {kind}")


def fit_model(spec: dict[str, Any], texts: list[str], y: np.ndarray) -> Any:
    """Entrena; si el candidato es D1 agrega LIAR-train con peso de dominio controlado."""
    model = build_model(spec)
    if spec.get("d1"):
        liar = liar_split("train", spec["profile"])
        w_isot = np.ones(len(texts))
        w_liar = np.full(len(liar), spec["d1_weight"] * len(texts) / max(1, len(liar)))
        texts = list(texts) + liar["doc"].tolist()
        y = np.r_[y, liar["label"].to_numpy()]
        model.fit(texts, y, clf__sample_weight=np.r_[w_isot, w_liar])
        return model
    model.fit(texts, y)
    return model


# --- Vistas ----------------------------------------------------------------------------------
def group_folds(pool: pd.DataFrame) -> list[dict[str, Any]]:
    """Folds por grupo: cada uno retiene un subject real y un grupo de subjects falsos."""
    small = ["left-news", "Government News", "US_News", "Middle-east"]
    plan = [("politicsNews", ["News"]), ("worldnews", ["politics"]),
            ("politicsNews", small), ("worldnews", small)]
    folds = []
    for real, fakes in plan:
        held = pool["subject"].isin([real, *fakes])
        folds.append({"nombre": f"{real} + {'/'.join(fakes) if len(fakes) < 3 else 'subjects falsos chicos'}",
                      "train_idx": np.where(~held)[0], "eval_idx": np.where(held)[0]})
    return folds


def temporal_split(pool: pd.DataFrame) -> dict[str, Any]:
    """Entrena con fechas anteriores al corte (config) y evalúa con las posteriores."""
    cut = pd.Timestamp(load_config()["panel"]["temporal_cutoff"])
    dt = pd.to_datetime(pool["date"].str.strip(), errors="coerce", format="mixed")
    ok = dt.notna().to_numpy()
    return {"corte": str(cut.date()), "train_idx": np.where(ok & (dt < cut).to_numpy())[0],
            "eval_idx": np.where(ok & (dt >= cut).to_numpy())[0]}


def view_metrics(y: np.ndarray, p: np.ndarray, y_fit: np.ndarray) -> dict[str, Any]:
    """Métricas de una vista + el trivial (clase mayoritaria del conjunto de ajuste)."""
    m = classification_metrics(y, (p >= 0.5).astype(int), p)
    maj = int(np.mean(y_fit) >= 0.5)
    triv = classification_metrics(y, np.full(len(y), maj), np.full(len(y), float(np.mean(y_fit))))
    return {"n": int(len(y)), "prop_falsas": float(np.mean(y)), "accuracy": m["accuracy"],
            "f1_macro": m["macro_f1"], "f1_falsa": m["fake"]["f1"], "f1_real": m["real"]["f1"],
            "brier": m["brier"], "ece": m["ece"], "trivial_accuracy": triv["accuracy"],
            "trivial_f1_macro": triv["macro_f1"], "trivial_brier": triv["brier"]}


def invariance_inputs(profile: str, n: int, seed: int) -> dict[str, list[str]]:
    """Textos de validación (muestra estratificada) transformados y limpiados con el perfil."""
    cache = get_paths().interim / f"invariance_{profile}_{n}.joblib"
    if cache.exists():
        return joblib.load(cache)
    va = isot_split("val", "v1")
    sample = va.groupby("label", group_keys=False).sample(frac=min(1.0, n / len(va)), random_state=seed)
    raw = isot_raw(sample["id"])
    out = {"__ids__": sample["id"].tolist(), "__y__": sample["label"].tolist(),
           "base": [build_document(t, b, profile=profile) for t, b in zip(raw["title"], raw["text"])]}
    for name, fn in TRANSFORMS.items():
        pairs = [fn(t, b) for t, b in zip(raw["title"], raw["text"])]
        out[name] = [build_document(t, b, profile=profile) for t, b in pairs]
    joblib.dump(out, cache)
    return out


def invariance_inputs_v3(profile: str, n: int, seed: int) -> dict[str, list[str]]:
    """Como `invariance_inputs`, con todas las transformaciones de la iteración 3."""
    cache = get_paths().interim / f"invariance_v3_{profile}_{n}.joblib"
    if cache.exists():
        return joblib.load(cache)
    va = isot_split("val", "v1")
    sample = va.groupby("label", group_keys=False).sample(frac=min(1.0, n / len(va)), random_state=seed)
    raw = isot_raw(sample["id"])
    out = {"__ids__": sample["id"].tolist(), "__y__": sample["label"].tolist(),
           "base": [build_document(t, b, profile=profile) for t, b in zip(raw["title"], raw["text"])]}
    for name, fn in TRANSFORMS_V3.items():
        out[name] = [build_document(*fn(t, b), profile=profile) for t, b in zip(raw["title"], raw["text"])]
    joblib.dump(out, cache)
    return out


def evaluate_candidate(spec: dict[str, Any], rigorous: bool = False) -> dict[str, Any]:
    """Corre todas las vistas para un candidato.

    Con ``rigorous=True`` (iteración 3) la invarianza usa todas las transformaciones de formato y
    el piso se aplica a la cota superior del IC 95 % de la tasa de cambio de cada transformación.
    """
    cfg = load_config()
    t0 = time.time()
    prof = spec["profile"]
    tr, va = isot_split("train", prof), isot_split("val", prof)
    pool = pd.concat([tr, va], ignore_index=True)
    out: dict[str, Any] = {"candidato": spec["name"], "spec": spec}
    # 1. in-domain + 4 + 4b + 5 usan el mismo modelo entrenado en ISOT train
    m = fit_model(spec, tr["doc"].tolist(), tr["label"].to_numpy())
    ytr = tr["label"].to_numpy()
    out["in_domain"] = view_metrics(va["label"].to_numpy(), M.p_fake(m, va["doc"].tolist()), ytr)
    wel = welfake_split("dev", prof)
    out["externo_dev_welfake"] = view_metrics(wel["label"].to_numpy(), M.p_fake(m, wel["doc"].tolist()), ytr)
    lv = liar_split("valid", prof)
    out["liar_valid"] = view_metrics(lv["label"].to_numpy(), M.p_fake(m, lv["doc"].tolist()), ytr)
    out["liar_valid"]["nota"] = "en D1 LIAR no es externo (se entrenó con LIAR-train)" if spec.get("d1") else ""
    if rigorous:
        inv = invariance_inputs_v3(prof, cfg["panel"]["invariance_sample"], cfg["seed"])
        p_base = M.p_fake(m, inv["base"])
        out["invarianza"] = {name: flip_stats_ci(p_base, M.p_fake(m, inv[name]), 0.5, cfg["seed"])
                             for name in TRANSFORMS_V3}
        out["invarianza_max_cambio"] = max(v["ic95_sup"] for v in out["invarianza"].values())
    else:
        inv = invariance_inputs(prof, cfg["panel"]["invariance_sample"], cfg["seed"])
        p_base = M.p_fake(m, inv["base"])
        out["invarianza"] = {name: flip_stats(p_base, M.p_fake(m, inv[name])) for name in TRANSFORMS}
        out["invarianza_max_cambio"] = max(v["tasa_cambio"] for v in out["invarianza"].values())
    # 2. grupo
    folds = []
    for f in group_folds(pool):
        fit = pool.iloc[f["train_idx"]]
        ev = pool.iloc[f["eval_idx"]]
        mf = fit_model(spec, fit["doc"].tolist(), fit["label"].to_numpy())
        folds.append({"fold": f["nombre"], **view_metrics(ev["label"].to_numpy(), M.p_fake(mf, ev["doc"].tolist()),
                                                          fit["label"].to_numpy())})
    out["grupo_folds"] = folds
    out["grupo"] = {k: float(np.mean([x[k] for x in folds])) for k in folds[0] if isinstance(folds[0][k], float)}
    # 3. tiempo
    ts = temporal_split(pool)
    fit, ev = pool.iloc[ts["train_idx"]], pool.iloc[ts["eval_idx"]]
    mt = fit_model(spec, fit["doc"].tolist(), fit["label"].to_numpy())
    out["tiempo"] = {"corte": ts["corte"], **view_metrics(ev["label"].to_numpy(), M.p_fake(mt, ev["doc"].tolist()),
                                                          fit["label"].to_numpy())}
    out["puntaje_robusto"] = float(np.mean([out["grupo"]["f1_macro"], out["tiempo"]["f1_macro"],
                                            out["externo_dev_welfake"]["f1_macro"]]))
    pc = cfg["panel"]
    out["piso_in_domain_ok"] = min(out["in_domain"]["f1_falsa"], out["in_domain"]["f1_real"]) >= pc["min_in_domain_f1"]
    out["piso_invarianza_ok"] = out["invarianza_max_cambio"] <= pc["max_invariance_flip"]
    out["elegible"] = bool(out["piso_in_domain_ok"] and out["piso_invarianza_ok"])
    out["segundos"] = round(time.time() - t0, 1)
    LOGGER.info("%s: robusto=%.4f in=%.4f grupo=%.4f tiempo=%.4f welfake=%.4f inv=%.3f (%.0fs)", spec["name"],
                out["puntaje_robusto"], out["in_domain"]["f1_macro"], out["grupo"]["f1_macro"],
                out["tiempo"]["f1_macro"], out["externo_dev_welfake"]["f1_macro"],
                out["invarianza_max_cambio"], out["segundos"])
    return out


def select_robust(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aplica el criterio de selección (igual para todos los candidatos)."""
    tie = load_config()["panel"]["tie_tolerance"]
    elig = [r for r in results if r["elegible"]]
    pool = elig or results
    best = max(r["puntaje_robusto"] for r in pool)
    tied = [r for r in pool if r["puntaje_robusto"] >= best - tie]
    chosen = min(tied, key=lambda r: (COMPLEXITY[r["spec"]["model"]], bool(r["spec"].get("d1")),
                                      -r["puntaje_robusto"]))
    return {"elegido": chosen["candidato"], "spec": chosen["spec"], "hubo_elegibles": bool(elig),
            "empatados": [r["candidato"] for r in tied],
            "criterio": "máx. promedio F1 macro (grupo, tiempo, WELFake-dev) con F1 por clase in-domain >= "
                        f"{load_config()['panel']['min_in_domain_f1']} e invarianza <= "
                        f"{load_config()['panel']['max_invariance_flip']:.0%}; empates (< {tie}) -> más simple"}


def flat_row(r: dict[str, Any]) -> dict[str, Any]:
    """Fila plana para la tabla del panel."""
    s = r["spec"]
    return {"candidato": r["candidato"], "perfil": s["profile"], "modelo": s["model"], "C": s.get("C"),
            "D1_liar_train": bool(s.get("d1")), "peso_D1": s.get("d1_weight"),
            "in_f1_falsa": r["in_domain"]["f1_falsa"], "in_f1_real": r["in_domain"]["f1_real"],
            "in_f1_macro": r["in_domain"]["f1_macro"], "in_ece": r["in_domain"]["ece"],
            "grupo_f1_macro": r["grupo"]["f1_macro"], "grupo_ece": r["grupo"]["ece"],
            "tiempo_f1_macro": r["tiempo"]["f1_macro"], "tiempo_ece": r["tiempo"]["ece"],
            "welfake_dev_f1_macro": r["externo_dev_welfake"]["f1_macro"],
            "welfake_dev_accuracy": r["externo_dev_welfake"]["accuracy"],
            "welfake_dev_ece": r["externo_dev_welfake"]["ece"],
            "liar_valid_f1_macro": r["liar_valid"]["f1_macro"], "liar_valid_accuracy": r["liar_valid"]["accuracy"],
            "invarianza_max_cambio": r["invarianza_max_cambio"], "puntaje_robusto": r["puntaje_robusto"],
            "elegible": r["elegible"], "segundos": r["segundos"]}


def run_panel(only: list[str] | None = None, n_jobs: int = 3) -> dict[str, Any]:
    """Evalúa todos los candidatos de `config.yaml` (panel.candidates) y elige el modelo final."""
    from joblib import Parallel, delayed

    cfg, paths = load_config(), get_paths()
    specs = [c for c in cfg["panel"]["candidates"] if not only or c["name"] in only]
    # Precalienta cachés (textos por perfil e invarianza) en el proceso principal.
    for prof in sorted({s["profile"] for s in specs}):
        isot_split("train", prof)
        welfake_split("dev", prof)
        invariance_inputs(prof, cfg["panel"]["invariance_sample"], cfg["seed"])
    heavy = [s for s in specs if s["model"] in ("lr_char", "ensemble")]
    light = [s for s in specs if s not in heavy]
    results = Parallel(n_jobs=n_jobs)(delayed(evaluate_candidate)(s) for s in light)
    results += Parallel(n_jobs=max(1, n_jobs - 1))(delayed(evaluate_candidate)(s) for s in heavy)
    prev_path = paths.reports / "panel_v2.json"
    if only and prev_path.exists():  # modo incremental: combina con los resultados ya calculados
        from .config import read_json

        new = {r["candidato"] for r in results}
        results += [r for r in read_json(prev_path)["resultados"] if r["candidato"] not in new]
        specs = cfg["panel"]["candidates"]
    order = {s["name"]: i for i, s in enumerate(specs)}
    results = [r for r in results if r["candidato"] in order]
    results.sort(key=lambda r: order[r["candidato"]])
    sel = select_robust(results)
    table = pd.DataFrame([flat_row(r) for r in results])
    table.to_csv(paths.reports / "panel_v2.csv", index=False)
    import datetime as dt

    out = {"generado": dt.datetime.now().isoformat(timespec="seconds"), "seleccion": sel, "resultados": results,
           "vistas": {
        "grupo": [f["nombre"] for f in group_folds(pd.concat([isot_split("train", "v1"), isot_split("val", "v1")],
                                                            ignore_index=True))],
        "tiempo_corte": cfg["panel"]["temporal_cutoff"]}}
    write_json(out, paths.reports / "panel_v2.json")
    return out


def ablation_report() -> pd.DataFrame:
    """Escalera de ablación v1 -> v2 -> v3 por familia de modelo (desde reports/panel_v2.csv)."""
    from . import plots

    paths = get_paths()
    d = pd.read_csv(paths.reports / "panel_v2.csv")
    d["familia"] = d["modelo"] + np.where(d["D1_liar_train"], "+D1(w=" + d["peso_D1"].astype(str) + ")", "") + \
        np.where(d["modelo"] == "lr_word", " C=" + d["C"].astype(str), "")
    cols = ["in_f1_macro", "grupo_f1_macro", "tiempo_f1_macro", "welfake_dev_f1_macro", "liar_valid_f1_macro",
            "invarianza_max_cambio", "puntaje_robusto"]
    table = d[["familia", "perfil", *cols]].sort_values(["familia", "perfil"])
    table.to_csv(paths.reports / "ablation_v2.csv", index=False)
    fams = [f for f in table["familia"].unique() if (table["familia"] == f).sum() > 1]
    vals = {prof: [float(table[(table.familia == f) & (table.perfil == prof)]["puntaje_robusto"].iloc[0])
                   if ((table.familia == f) & (table.perfil == prof)).any() else 0.0 for f in fams]
            for prof in ("v1", "v2", "v3")}
    plots.bar_compare(fams, {f"Perfil {k}": v for k, v in vals.items()},
                      "Escalera de ablación: puntaje robusto por perfil de limpieza", "Puntaje robusto (F1 macro)",
                      paths.figures / "ablation_v2.png", (0.5, 0.8),
                      colors=[plots.NEUTRAL, plots.SERIES[6], plots.SERIES[3]])
    return table


def run_panel_v3(only: list[str] | None = None, n_jobs: int = 2) -> dict[str, Any]:
    """Panel de la iteración 3: conjunto chico de candidatos, invarianza rigurosa y regla de aceptación.

    Escribe `reports/panel_v3.*` (no toca `panel_v2.*`). Regla de aceptación: el ganador con `v2_1` no
    puede perder más de `acceptance_max_drop` de F1 macro en ninguna vista frente al modelo de la
    iteración 2 medido en esta misma corrida, ni empeorar la invarianza.
    """
    import datetime as dt

    from joblib import Parallel, delayed

    from .config import read_json

    cfg, paths = load_config(), get_paths()
    pc = cfg["panel_v3"]
    specs = [c for c in pc["candidates"] if not only or c["name"] in only]
    for prof in sorted({s["profile"] for s in specs}):
        isot_split("train", prof)
        welfake_split("dev", prof)
        invariance_inputs_v3(prof, cfg["panel"]["invariance_sample"], cfg["seed"])
    results = Parallel(n_jobs=n_jobs)(delayed(evaluate_candidate)(s, True) for s in specs)
    prev = paths.reports / "panel_v3.json"
    if only and prev.exists():
        new = {r["candidato"] for r in results}
        results += [r for r in read_json(prev)["resultados"] if r["candidato"] not in new]
    order = {s["name"]: i for i, s in enumerate(pc["candidates"])}
    results = sorted([r for r in results if r["candidato"] in order], key=lambda r: order[r["candidato"]])
    ref = next(r for r in results if r["candidato"] == pc["reference"])
    views = {"in_domain": "in_domain", "grupo": "grupo", "tiempo": "tiempo", "externo_dev_welfake": "externo_dev_welfake"}
    for r in results:
        drops = {v: ref[k]["f1_macro"] - r[k]["f1_macro"] for v, k in views.items()}
        r["caidas_vs_iteracion_2"] = drops
        r["aceptado"] = bool(max(drops.values()) <= pc["acceptance_max_drop"]
                             and r["invarianza_max_cambio"] <= max(ref["invarianza_max_cambio"], cfg["panel"]["max_invariance_flip"]))
    pool = [r for r in results if r["candidato"] != pc["reference"] and r["spec"]["model"] != "emb_lr"]
    sel = select_robust([r for r in pool if r["aceptado"]] or pool)
    sel["hubo_aceptados"] = any(r["aceptado"] for r in pool)
    sel["decision_final"] = sel["elegido"] if sel["hubo_aceptados"] else pc["reference"]
    sel["motivo_decision"] = ("el mejor candidato cumple la regla de aceptación" if sel["hubo_aceptados"] else
                              "ningún candidato v2_1 cumple la regla de aceptación: se conserva el clasificador de la "
                              "iteración 2 con las correcciones A1 y B (prompt, sección C)")
    emb = next((r for r in results if r["spec"]["model"] == "emb_lr"), None)
    if emb is not None:
        best = max(r["puntaje_robusto"] for r in results if r["spec"]["model"] != "emb_lr")
        best_clean = max(r["puntaje_robusto"] for r in pool)
        sel["embeddings"] = {"puntaje_robusto": emb["puntaje_robusto"], "mejor_sin_embeddings": best,
                             "mejor_sin_embeddings_con_v2_1": best_clean, "mejora": emb["puntaje_robusto"] - best,
                             "mejora_vs_v2_1": emb["puntaje_robusto"] - best_clean,
                             "adoptar": bool(emb["puntaje_robusto"] - best >= pc["embedding_min_gain"] and emb["elegible"])}
    rows = []
    for r in results:
        row = flat_row(r)
        row["invarianza_peor_transformacion"] = max(r["invarianza"], key=lambda k: r["invarianza"][k]["ic95_sup"])
        row["aceptado_vs_it2"] = r["aceptado"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(paths.reports / "panel_v3.csv", index=False)
    out = {"generado": dt.datetime.now().isoformat(timespec="seconds"), "seleccion": sel, "resultados": results}
    write_json(out, paths.reports / "panel_v3.json")
    return out
