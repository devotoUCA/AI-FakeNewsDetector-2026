"""Calibración, invarianza con IC, compuerta, log de accesos y consistencia cifras-documentos."""

import json
import re

import pytest

from fakenews.access_log import read_log, verify_access_log
from fakenews.config import get_paths, load_config, read_json
from fakenews.make_report import TABLES, load_all, replace_blocks

from .conftest_v3 import needs_v3_reports


@needs_v3_reports
def test_in_domain_ece_after_fix():
    r = read_json(get_paths().reports / "shrink_resolution_v3.json")
    assert r["isot_val"]["it3_calibrada_en_dominio"]["ece"] <= 0.02
    assert r["isot_val"]["it3_calibrada_todos"]["ece"] <= 0.02
    assert r["isot_val"]["it2_contraccion_global"]["ece"] > 0.1  # el defecto que se corrigió


@needs_v3_reports
def test_invariance_report_has_ci_for_every_transform():
    from fakenews.invariance import TRANSFORMS_V3

    inv = read_json(get_paths().reports / "invariance_v3.json")
    assert inv["n"] >= 2000 and set(inv["transformaciones"]) == set(TRANSFORMS_V3)
    for v in inv["transformaciones"].values():
        assert 0 <= v["ic95_inf"] <= v["tasa_cambio"] <= v["ic95_sup"] <= 1


@needs_v3_reports
@pytest.mark.xfail(strict=True, reason="CONOCIDO Y DECLARADO: el modelo final (clasificador de la iteración 2) no "
                   "cumple el criterio en 'oraciones_pegadas'. Ver docs/informe_resultados.md. No se relaja el umbral.")
def test_final_model_meets_invariance_criterion():
    inv = read_json(get_paths().reports / "invariance_v3.json")
    assert inv["cumple_todas"], f"No cumplen: {inv['no_cumplen']}"


@needs_v3_reports
def test_invariance_failures_are_declared_in_documents():
    inv = read_json(get_paths().reports / "invariance_v3.json")
    root = get_paths().root
    for rel in ("docs/informe_resultados.md", "README.md", "docs/model_card.md"):
        text = (root / rel).read_text(encoding="utf-8").replace(" ", "_")
        for name in inv["no_cumplen"]:
            assert name in text, f"{rel} no declara el incumplimiento de {name}"


@needs_v3_reports
def test_domain_gate_report():
    g = read_json(get_paths().reports / "domain_gate_v3.json")
    best = g["detectores"][g["elegido"]]
    assert best["auc_pertenencia"] > 0.5 and {"sin_ver_WELFake", "sin_ver_LIAR"} == set(g["dejar_un_dominio_afuera"])
    keep = next(r for r in best["por_retencion"] if r["retencion"] == load_config()["domain_gate"]["min_in_domain_retention"])
    assert keep["retencion_real"] >= 0.965


def test_log_has_retroactive_entries_without_fake_dates():
    log = read_log()
    retro = [e for e in log if e.get("retroactive")]
    assert retro and all(e.get("fuente") and e.get("fecha_evento") for e in retro)
    assert {"liar_train+valid+test", "isot_test"} <= {e["conjunto"] for e in retro}
    earliest = min(e["ts"] for e in log)
    assert all(e["ts"] > earliest for e in retro)  # registradas después: no se falsificó la fecha


@needs_v3_reports
def test_second_look_is_declared_and_virgin_holdout_touched_once():
    it3 = [e for e in read_log() if e.get("iteracion") == 3 and not e.get("retroactive")]
    # La única repetición permitida es la recomputación técnica declarada (bug en el análisis de errores).
    repeated = [e for e in it3 if e["reacceso"]]
    assert all("recomputación técnica" in e["motivo"] for e in repeated)
    first = [e for e in it3 if not e["reacceso"]]
    motivos = {e["conjunto"]: e["motivo"] for e in first}
    assert "segunda mirada declarada" in motivos["isot_test"] and "segunda mirada declarada" in motivos["welfake_holdout"]
    fk = [e for e in it3 if e["conjunto"] == "fakes_holdout"]
    assert len({e["modelo"] for e in fk}) == len(fk) == 3 and not any(e["reacceso"] for e in fk)


@needs_v3_reports
def test_verify_access_log_passes():
    r = verify_access_log()
    assert r["ok"], r["problemas"]
    assert r["cifras_revisadas"] > 10


@needs_v3_reports
def test_verifier_detects_invented_figure(tmp_path):
    (tmp_path / "README.md").write_text("WELFake holdout: accuracy 0,987654 y 0,321\n", encoding="utf-8")
    r = verify_access_log(root=tmp_path)
    assert not r["ok"] and any("0,321" in p for p in r["problemas"])


@needs_v3_reports
def test_generated_tables_match_reports():
    root = get_paths().root
    d = load_all()
    tables = {k: f(d) for k, f in TABLES.items()}
    for rel in ("README.md", "docs/informe_resultados.md", "docs/model_card.md", "docs/campos_completar.md"):
        text = (root / rel).read_text(encoding="utf-8")
        assert "BEGIN GENERATED" in text, rel
        assert replace_blocks(text, tables) == text, f"{rel} está desactualizado: correr `python -m fakenews.cli report`"


@needs_v3_reports
def test_key_figures_in_readme_exist_in_reports():
    text = (get_paths().root / "README.md").read_text(encoding="utf-8")
    blob = json.dumps(read_json(get_paths().reports / "final_v3.json"))
    acc = read_json(get_paths().reports / "final_v3.json")["v3"]["isot_test"]["accuracy"]
    assert f"{acc:.3f}".replace(".", ",") in text and str(acc) in blob
    assert not re.search(r"\[completar\]", text)


def test_error_files_of_previous_iterations_untouched():
    import subprocess

    root = get_paths().root
    inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root, capture_output=True, check=False)
    if inside.returncode != 0:
        pytest.skip("sin repo git (copia limpia)")
    for name in ("reports/error_analysis.csv", "reports/error_analysis_v2.csv"):
        r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", name], cwd=root, capture_output=True, check=False)
        assert r.returncode == 0, name


@needs_v3_reports
def test_error_analysis_v3_strata():
    import pandas as pd

    d = pd.read_csv(get_paths().reports / "error_analysis_v3.csv", keep_default_na=False)
    assert len(d) >= 40 and d["comentario_borrador"].str.startswith("BORRADOR").all()
    # Las columnas de revisión las completa el equipo: deben existir, y nadie firma sin comentar.
    assert {"comentario_equipo", "revisado_por"} <= set(d.columns)
    assert ((d["revisado_por"].str.strip() == "") | (d["comentario_equipo"].str.strip() != "")).all()
    assert d["origen"].nunique() == 2 and d["tipo"].nunique() == 2 and d["estrato_confianza"].nunique() >= 3
    assert (d["estrato_confianza"] == "fuera_de_dominio").sum() >= 1
