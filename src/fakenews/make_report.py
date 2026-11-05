"""Genera las tablas y cifras de los documentos desde `reports/` (nada se tipea a mano).

En cada documento, el bloque entre `<!-- BEGIN GENERATED:nombre -->` y `<!-- END GENERATED:nombre -->`
se reemplaza por la tabla `nombre`. También escribe `reports/holdout_figures_v3.json` (registro de
las cifras de conjuntos reservados, lo usa `verify_access_log`) y `reports/comparison_it1_it2_it3.csv`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import get_paths, read_json, write_json

DOCS = ("README.md", "docs/informe_resultados.md", "docs/model_card.md", "docs/campos_completar.md",
        "docs/guion_defensa.md", "docs/02_iteracion_3.md", "docs/data_card.md")
PRETTY = {"dummy": "Trivial (clase mayoritaria)", "logreg_raw": "LR palabras (baseline)", "naive_bayes_cal": "Naive Bayes",
          "linear_svm_cal": "SVM lineal (final it. 1)", "style_tree": "Árbol de estilo", "style_forest": "Bosque de estilo",
          "ensemble": "Ensamble", "knn": "kNN coseno", "mlp": "Red pequeña (MLP)"}
NAMES = {"isot_test": "ISOT test", "welfake_holdout": "WELFake holdout", "liar_test": "LIAR test (no virgen)",
         "fakes_holdout": "FA-KES (virgen)", "isot_val": "ISOT val", "welfake_dev": "WELFake dev", "liar_valid": "LIAR valid"}


def n(x: float | None, d: int = 3) -> str:
    """Número con coma decimal."""
    return "—" if x is None else f"{x:.{d}f}".replace(".", ",")


def pct(x: float | None, d: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{d}f} %".replace(".", ",")


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load_all() -> dict[str, Any]:
    r = get_paths().reports
    return {"final": read_json(r / "final_v3.json"), "f2": read_json(r / "final_v2.json"), "p2": read_json(r / "panel_v2.json"),
            "p3": read_json(r / "panel_v3.json"), "t1": read_json(r / "test_results.json"), "liar1": read_json(r / "external_liar.json"),
            "gate": read_json(r / "domain_gate_v3.json"), "inv": read_json(r / "invariance_v3.json"),
            "shrink": read_json(r / "shrink_resolution_v3.json"), "train": read_json(r / "train_v3.json"),
            "crit": read_json(r / "success_criteria_v3.json")}


def _panel(d: dict[str, Any], results: str, name: str) -> dict[str, Any]:
    return next(x for x in d[results]["resultados"] if x["candidato"] == name)


def comparison(d: dict[str, Any]) -> pd.DataFrame:
    """Iteraciones 1, 2 y 3 en las vistas del panel y en los holdouts."""
    a, b = _panel(d, "p2", "it1_svm_v1"), _panel(d, "p2", "lrchar_v2_C10")
    c = _panel(d, "p3", d["final"]["modelo"])
    f, f2 = d["final"]["v3"], d["f2"]
    t1 = d["t1"]["modelos"]["linear_svm_cal"]
    rows = [("ISOT val, F1 macro", a["in_domain"]["f1_macro"], b["in_domain"]["f1_macro"], c["in_domain"]["f1_macro"]),
            ("Cambio de subject, F1 macro", a["grupo"]["f1_macro"], b["grupo"]["f1_macro"], c["grupo"]["f1_macro"]),
            ("Cambio temporal, F1 macro", a["tiempo"]["f1_macro"], b["tiempo"]["f1_macro"], c["tiempo"]["f1_macro"]),
            ("WELFake dev, F1 macro", a["externo_dev_welfake"]["f1_macro"], b["externo_dev_welfake"]["f1_macro"], c["externo_dev_welfake"]["f1_macro"]),
            ("LIAR valid, F1 macro", a["liar_valid"]["f1_macro"], b["liar_valid"]["f1_macro"], c["liar_valid"]["f1_macro"]),
            ("Puntaje robusto", a["puntaje_robusto"], b["puntaje_robusto"], c["puntaje_robusto"]),
            ("ISOT test, accuracy", t1["accuracy"], f2["v2"]["isot_test"]["forzado"]["accuracy"], f["isot_test"]["accuracy"]),
            ("ISOT test, F1 falsa", t1["fake"]["f1"], f2["v2"]["isot_test"]["forzado"]["f1_fake"], f["isot_test"]["f1_fake"]),
            ("WELFake holdout, accuracy", f2["it1"]["welfake_holdout"]["forzado"]["accuracy"], f2["v2"]["welfake_holdout"]["forzado"]["accuracy"], f["welfake_holdout"]["accuracy"]),
            ("WELFake holdout, F1 macro", f2["it1"]["welfake_holdout"]["forzado"]["macro_f1"], f2["v2"]["welfake_holdout"]["forzado"]["macro_f1"], f["welfake_holdout"]["macro_f1"]),
            ("LIAR test, accuracy", f2["it1"]["liar_test_no_virgen"]["forzado"]["accuracy"], f2["v2"]["liar_test_no_virgen"]["forzado"]["accuracy"], f["liar_test"]["accuracy"]),
            ("FA-KES, accuracy", d["final"]["it1_fakes_holdout"]["accuracy"], d["final"]["it2_fakes_holdout"]["accuracy"], f["fakes_holdout"]["accuracy"]),
            ("FA-KES, F1 macro", d["final"]["it1_fakes_holdout"]["macro_f1"], d["final"]["it2_fakes_holdout"]["macro_f1"], f["fakes_holdout"]["macro_f1"])]
    return pd.DataFrame(rows, columns=["vista", "iteracion_1", "iteracion_2", "iteracion_3"])


def t_iteraciones(d: dict[str, Any]) -> str:
    c = comparison(d)
    f = d["final"]["v3"]
    triv = {"WELFake holdout, accuracy": f["welfake_holdout"]["trivial"]["accuracy"], "LIAR test, accuracy": f["liar_test"]["trivial"]["accuracy"],
            "FA-KES, accuracy": f["fakes_holdout"]["trivial"]["accuracy"], "ISOT test, accuracy": f["isot_test"]["trivial"]["accuracy"]}
    return table(["Vista", "It. 1", "It. 2", "It. 3", "Trivial"],
                 [[r.vista, n(r.iteracion_1), n(r.iteracion_2), n(r.iteracion_3), n(triv.get(r.vista))] for r in c.itertuples()])


def t_acta(d: dict[str, Any]) -> str:
    a = pd.read_csv(get_paths().reports / "acta_comparison_v3.csv")
    rows = []
    for r in a.itertuples():
        name = PRETTY.get(r.modelo, r.modelo)
        ic = str(r.ic95_f1_fake).strip("[]").replace(",", " –").replace(".", ",")
        rows.append([name, r.limpieza, n(r.accuracy), n(r.precision_fake), n(r.recall_fake), f"{n(r.f1_fake)} [{ic}]", n(r.precision_real),
                     n(r.recall_real), n(r.f1_real), n(r.brier, 4), n(r.log_loss)])
    return table(["Modelo", "Limpieza", "Accuracy", "Prec. falsa", "Rec. falsa", "F1 falsa [IC 95 %]", "Prec. real", "Rec. real", "F1 real", "Brier", "Log-loss"], rows)


def t_mcnemar(d: dict[str, Any]) -> str:
    m = d["final"]["mcnemar_baseline_vs_final"]
    return table(["Solo acierta el baseline", "Solo acierta el modelo final", "p (McNemar exacto)"],
                 [[m["solo_A_acierta"], m["solo_B_acierta"], n(m["p_value"], 4)]])


def t_holdouts(d: dict[str, Any]) -> str:
    rows = []
    for k in ("isot_test", "welfake_holdout", "liar_test", "fakes_holdout"):
        m = d["final"]["v3"][k]
        ic = m["ic95"]["accuracy"]
        rows.append([NAMES[k], m["n"], f"{n(m['accuracy'])} [{n(ic[0])}–{n(ic[1])}]", n(m["trivial"]["accuracy"]), n(m["macro_f1"]), n(m["f1_fake"]), n(m["f1_real"]), n(m["ece"]), n(m["brier"])])
    return table(["Conjunto", "n", "Accuracy [IC 95 %]", "Trivial", "F1 macro", "F1 falsa", "F1 real", "ECE", "Brier"], rows)


def t_dominio(d: dict[str, Any]) -> str:
    rows = []
    for k, label in (("isot_val", "ISOT val"), ("welfake_dev", "WELFake dev"), ("liar_valid", "LIAR valid")):
        m = d["train"]["dominio_dev"][k]
        rows.append([label, pct(m["frac_fuera_de_dominio"]), m["n_en_dominio"], n(m["accuracy_en_dominio"]), n(m["trivial_en_dominio"])])
    for k in ("isot_test", "welfake_holdout", "liar_test", "fakes_holdout"):
        m = d["final"]["dominio"][k]
        rows.append([NAMES[k], pct(m["frac_fuera_de_dominio"]), m["n_en_dominio"], n(m["accuracy_en_dominio"]), n(m["trivial_en_dominio"])])
    return table(["Conjunto", "Fuera de dominio", "Responde (n)", "Accuracy sobre lo respondido", "Trivial sobre lo mismo"], rows)


def t_detectores(d: dict[str, Any]) -> str:
    rows = []
    for k, v in d["gate"]["detectores"].items():
        r = {x["retencion"]: x for x in v["por_retencion"]}
        rows.append([k + (" (elegido)" if k == d["gate"]["elegido"] else ""), n(v["auc_pertenencia"]), n(v["auc_vs_welfake"]), n(v["auc_vs_liar"]),
                     pct(r[0.95]["deteccion_fuera_de_dominio"]), pct(r[0.97]["deteccion_fuera_de_dominio"]), pct(r[0.99]["deteccion_fuera_de_dominio"])])
    return table(["Detector", "AUC pertenencia", "AUC vs. WELFake", "AUC vs. LIAR", "Detección a 95 % ret.", "a 97 %", "a 99 %"], rows)


def t_lodo(d: dict[str, Any]) -> str:
    rows = [[k.replace("sin_ver_", "Sin ver "), n(v["auc"]), pct(v["deteccion_fuera_de_dominio"])] for k, v in d["gate"]["dejar_un_dominio_afuera"].items()]
    return table(["Entrenada", "AUC", "Detección a 97 % de retención"], rows)


def t_invarianza(d: dict[str, Any]) -> str:
    rows = [[k.replace("_", " "), pct(v["tasa_cambio"], 2), f"[{pct(v['ic95_inf'], 2)} – {pct(v['ic95_sup'], 2)}]",
             "sí" if v["ic95_sup"] <= d["inv"]["limite_cota_superior"] else "**no**"] for k, v in d["inv"]["transformaciones"].items()]
    return table(["Transformación", "Tasa de cambio", "IC 95 %", "Cota superior ≤ 2 %"], rows)


def t_calibracion(d: dict[str, Any]) -> str:
    rows = []
    for k in ("isot_val", "welfake_dev", "liar_valid"):
        s = d["shrink"][k]
        a, b, c = s["it2_contraccion_global"], s["it3_calibrada_todos"], s["it3_calibrada_en_dominio"]
        rows.append([NAMES[k], n(a["ece"]), n(a["brier"]), n(b["ece"]), n(b["brier"]), n(c["ece"]) if c else "—", n(c["brier"]) if c else "—"])
    return table(["Conjunto", "ECE it. 2 (contracción)", "Brier it. 2", "ECE it. 3 (calibrada)", "Brier it. 3", "ECE it. 3 en dominio", "Brier it. 3 en dominio"], rows)


def t_panel_v3(d: dict[str, Any]) -> str:
    rows = []
    for r in d["p3"]["resultados"]:
        s = r["spec"]
        rows.append([r["candidato"], s["profile"], n(r["in_domain"]["f1_falsa"]), n(r["grupo"]["f1_macro"]), n(r["tiempo"]["f1_macro"]), n(r["externo_dev_welfake"]["f1_macro"]),
                     pct(r["invarianza_max_cambio"], 2), n(r["puntaje_robusto"]), "sí" if r["aceptado"] else "no"])
    return table(["Candidato", "Perfil", "In-domain F1 falsa", "Subject", "Tiempo", "WELFake dev", "Invarianza (peor cota sup.)", "Robusto", "Aceptado"], rows)


def t_criterios(d: dict[str, Any]) -> str:
    return table(["Criterio del acta", "Estado"], [[c["criterio"], c["estado"]] for c in d["crit"]["criterios"]])


def t_readme(d: dict[str, Any]) -> str:
    f = d["final"]["v3"]
    dom = d["final"]["dominio"]
    rows = [[NAMES[k], n(f[k]["accuracy"]), n(f[k]["trivial"]["accuracy"]), pct(dom[k]["frac_fuera_de_dominio"]), n(dom[k]["accuracy_en_dominio"]), n(dom[k]["trivial_en_dominio"])]
            for k in ("isot_test", "welfake_holdout", "liar_test", "fakes_holdout")]
    return table(["Conjunto", "Accuracy (forzando respuesta)", "Trivial", "Fuera de dominio", "Accuracy sobre lo respondido", "Trivial sobre lo mismo"], rows)


TABLES = {"iteraciones": t_iteraciones, "acta": t_acta, "mcnemar": t_mcnemar, "holdouts": t_holdouts, "dominio": t_dominio,
          "detectores": t_detectores, "dejar_uno_afuera": t_lodo, "invarianza": t_invarianza, "calibracion": t_calibracion,
          "panel_v3": t_panel_v3, "criterios": t_criterios, "readme": t_readme}


def _numbers(obj: Any) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)] if 0 <= obj <= 1 else []
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _numbers(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in _numbers(v)]
    return []


def registry(d: dict[str, Any]) -> dict[str, Any]:
    """Cifras de conjuntos reservados, por conjunto e iteración."""
    f, f2 = d["final"], d["f2"]
    src = [("isot_test", 1, d["t1"]["modelos"]), ("isot_test", 2, f2["v2"]["isot_test"]), ("isot_test", 3, [f["v3"]["isot_test"], f["baseline_it1"], f["dominio"]["isot_test"], f["mcnemar_baseline_vs_final"]]),
           ("welfake_holdout", 2, [f2["v2"]["welfake_holdout"], f2["it1"]["welfake_holdout"]]), ("welfake_holdout", 3, [f["v3"]["welfake_holdout"], f["dominio"]["welfake_holdout"]]),
           ("liar_test", 1, d["liar1"]["modelos"]), ("liar_test", 2, [f2["v2"]["liar_test_no_virgen"], f2["it1"]["liar_test_no_virgen"]]),
           ("liar_test", 3, [f["v3"]["liar_test"], f["dominio"]["liar_test"]]),
           ("fakes_holdout", 3, [f["v3"]["fakes_holdout"], f["dominio"]["fakes_holdout"], f["it1_fakes_holdout"], f["it2_fakes_holdout"]])]
    items = []
    for conj, it, obj in src:
        for v in sorted(set(_numbers(obj))):
            items.append({"conjunto": conj, "iteracion": it, "valor": v})
    return {"cifras": items}


def replace_blocks(text: str, tables: dict[str, str]) -> str:
    def sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in tables:
            raise KeyError(f"Tabla generada desconocida: {name}")
        return f"<!-- BEGIN GENERATED:{name} -->\n{tables[name]}\n<!-- END GENERATED:{name} -->"

    return re.sub(r"<!-- BEGIN GENERATED:(\w+) -->.*?<!-- END GENERATED:\1 -->", sub, text, flags=re.S)


def make_report(root: Path | None = None) -> dict[str, Any]:
    """Regenera tablas, registro y comparación. Devuelve qué documentos cambió."""
    paths = get_paths()
    root = root or paths.root
    d = load_all()
    comparison(d).to_csv(paths.reports / "comparison_it1_it2_it3.csv", index=False)
    write_json(registry(d), paths.reports / "holdout_figures_v3.json")
    tables = {k: f(d) for k, f in TABLES.items()}
    changed = []
    for rel in DOCS:
        path = root / rel
        if not path.exists():
            continue
        old = path.read_text(encoding="utf-8")
        new = replace_blocks(old, tables)
        if new != old:
            path.write_text(new, encoding="utf-8")
            changed.append(rel)
    return {"tablas": sorted(tables), "documentos_actualizados": changed}
