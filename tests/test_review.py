"""Revisión humana de errores: qué cuenta como revisado y cómo se actualiza el criterio del acta."""

import json

import pandas as pd
import pytest

from fakenews import review
from fakenews.config import get_paths


def _csv(tmp_path, rows):
    base = {"id": "", "origen": "isot_test", "tipo": "FN (falsa dada por real)", "comentario_equipo": "", "revisado_por": ""}
    df = pd.DataFrame([base | r for r in rows])
    path = tmp_path / "error_analysis_v3.csv"
    df.to_csv(path, index=False)
    return path


def _rows(n_signed, n_unsigned=0, n_empty=0):
    rows = [{"id": f"s{i}", "comentario_equipo": "cita oficial republicada", "revisado_por": "Ana"} for i in range(n_signed)]
    rows += [{"id": f"u{i}", "comentario_equipo": "propuesta sin leer"} for i in range(n_unsigned)]
    rows += [{"id": f"e{i}"} for i in range(n_empty)]
    return rows


def test_comment_without_signature_does_not_count(tmp_path):
    df = review.load_errors(_csv(tmp_path, _rows(3, n_unsigned=30, n_empty=2)))
    s = review.review_summary(df)
    assert s["revisados"] == 3 and s["comentados_sin_firma"] == 30 and not s["cumple"]
    assert review.criterion_row(df)["estado"].startswith("pendiente")


def test_blank_spaces_are_not_a_signature(tmp_path):
    df = review.load_errors(_csv(tmp_path, [{"id": "a", "comentario_equipo": "algo", "revisado_por": "   "}]))
    assert review.count_reviewed(df) == 0


def test_twenty_signed_cases_meet_the_criterion(tmp_path):
    df = review.load_errors(_csv(tmp_path, _rows(19) + [{"id": "z", "comentario_equipo": "ok", "revisado_por": "Beto"}]))
    row = review.criterion_row(df)
    assert row["estado"] == "cumplido" and row["valor"]["comentados"] == 20
    assert row["valor"]["revisores"] == ["Ana", "Beto"]


def test_excel_bom_is_tolerated(tmp_path):
    path = _csv(tmp_path, _rows(2))
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert review.count_reviewed(review.load_errors(path)) == 2


def test_missing_columns_or_duplicated_ids_are_rejected(tmp_path):
    path = tmp_path / "error_analysis_v3.csv"
    pd.DataFrame([{"id": "a", "comentario_equipo": "x"}]).to_csv(path, index=False)
    with pytest.raises(review.ReviewFileError, match="revisado_por"):
        review.load_errors(path)
    with pytest.raises(review.ReviewFileError, match="repetidos"):
        review.load_errors(_csv(tmp_path, [{"id": "a"}, {"id": "a"}]))
    with pytest.raises(review.ReviewFileError, match="Falta"):
        review.load_errors(tmp_path / "no_existe.csv")


def test_update_only_touches_the_review_criterion(tmp_path):
    _csv(tmp_path, _rows(21))
    other = {"criterio": "Accuracy en test > 80 %", "valor": 0.99, "estado": "cumplido (solo dentro de ISOT)"}
    pending = {"criterio": ">= 20 errores revisados por el equipo", "valor": {"casos": 21, "comentados": 0}, "estado": review.PENDING}
    crit = tmp_path / "success_criteria_v3.json"
    crit.write_text(json.dumps({"modelo": "m", "criterios": [other, pending], "generado": "2026-11-05T19:00:44"}), encoding="utf-8")
    s = review.update_review_status(tmp_path)
    after = json.loads(crit.read_text(encoding="utf-8"))
    assert s["estado"] == "cumplido" and after["criterios"][1]["estado"] == "cumplido"
    assert after["criterios"][0] == other and after["generado"] == "2026-11-05T19:00:44" and after["modelo"] == "m"
    stamp = after["revision_actualizada"]
    review.update_review_status(tmp_path)  # sin cambios en el CSV no se reescribe nada
    assert json.loads(crit.read_text(encoding="utf-8"))["revision_actualizada"] == stamp


def test_regenerating_the_csv_keeps_team_reviews(tmp_path):
    old = _csv(tmp_path, [{"id": "a", "comentario_equipo": "leído", "revisado_por": "Ana"}, {"id": "b"}])
    new = pd.DataFrame([{"id": "b", "comentario_equipo": "", "revisado_por": ""},
                        {"id": "a", "comentario_equipo": "", "revisado_por": ""},
                        {"id": "c", "comentario_equipo": "", "revisado_por": ""}])
    out = review.carry_over_reviews(new, old).set_index("id")
    assert out.loc["a", "comentario_equipo"] == "leído" and out.loc["a", "revisado_por"] == "Ana"
    assert out.loc["b", "revisado_por"] == "" and out.loc["c", "comentario_equipo"] == ""
    assert review.carry_over_reviews(new, tmp_path / "no_existe.csv") is new


def test_repo_csv_and_criteria_are_consistent():
    """El estado publicado en `success_criteria_v3.json` coincide con lo que hay firmado en el CSV."""
    reports = get_paths().reports
    if not (reports / "error_analysis_v3.csv").exists() or not (reports / "success_criteria_v3.json").exists():
        pytest.skip("Faltan los reportes de la iteración 3.")
    df = review.load_errors(reports / "error_analysis_v3.csv")
    crit = json.loads((reports / "success_criteria_v3.json").read_text(encoding="utf-8"))
    row = next(c for c in crit["criterios"] if c["criterio"].startswith(review.CRITERION_PREFIX))
    assert row["estado"] == review.criterion_row(df)["estado"], "correr `python -m fakenews.cli report`"
    assert row["valor"]["comentados"] == review.count_reviewed(df), "correr `python -m fakenews.cli report`"
