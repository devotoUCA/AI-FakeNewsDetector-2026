"""Revisión humana de errores: cuenta los casos firmados y actualiza el criterio del acta.

El acta pide "al menos 20 ejemplos mal clasificados revisados y comentados". El equipo completa dos
columnas de `reports/error_analysis_v3.csv`:

- `comentario_equipo`: qué pasó en ese caso (puede partir de una propuesta, pero hay que leerla);
- `revisado_por`: nombre de la persona del equipo que leyó el caso y avala el comentario.

Un caso cuenta como revisado **solo si tiene las dos columnas completas**. Un comentario sin firma es
una propuesta, no una revisión.

Este módulo no carga modelos ni datos: solo lee el CSV ya generado. Por eso se puede correr las veces
que haga falta sin volver a mirar los conjuntos reservados (no agrega entradas al log de accesos).

    python -m fakenews.cli review-status
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

from .config import get_paths, read_json, write_json

REVIEW_COLUMNS = ("comentario_equipo", "revisado_por")
CRITERION_PREFIX = ">= 20 errores revisados"
MIN_REVIEWED = 20
PENDING = "pendiente: equipo (error_analysis_v3.csv)"


class ReviewFileError(ValueError):
    """El CSV de errores no tiene el formato esperado."""


def load_errors(path: Path) -> pd.DataFrame:
    """Lee el CSV de errores con las columnas de revisión como texto (vacío = sin completar)."""
    if not path.exists():
        raise ReviewFileError(f"Falta {path}. Se genera con `python -m fakenews.cli evaluate-v3`.")
    # utf-8-sig tolera el BOM que agrega Excel al guardar como "CSV UTF-8".
    df = pd.read_csv(path, keep_default_na=False, dtype=str, encoding="utf-8-sig")
    missing = [c for c in ("id", *REVIEW_COLUMNS) if c not in df.columns]
    if missing:
        raise ReviewFileError(
            f"{path.name}: faltan las columnas {missing}. Si se editó con Excel, guardar como "
            "'CSV UTF-8 (delimitado por comas)' sin borrar ni renombrar columnas.")
    if df["id"].duplicated().any():
        raise ReviewFileError(f"{path.name}: hay ids repetidos; no se deben duplicar filas.")
    return df


def signed_mask(df: pd.DataFrame) -> pd.Series:
    """Casos revisados: comentario y firma, los dos completos."""
    return (df["comentario_equipo"].str.strip() != "") & (df["revisado_por"].str.strip() != "")


def count_reviewed(df: pd.DataFrame) -> int:
    return int(signed_mask(df).sum())


def review_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Cuántos casos están firmados, cuántos solo comentados y cómo se reparten."""
    signed = signed_mask(df)
    commented = df["comentario_equipo"].str.strip() != ""
    out: dict[str, Any] = {
        "casos": int(len(df)),
        "revisados": int(signed.sum()),
        "comentados_sin_firma": int((commented & ~signed).sum()),
        "firma_sin_comentario": int(((df["revisado_por"].str.strip() != "") & ~commented).sum()),
        "minimo": MIN_REVIEWED,
        "cumple": bool(signed.sum() >= MIN_REVIEWED),
        "revisores": sorted({x.strip() for x in df.loc[signed, "revisado_por"]}),
    }
    for col in ("origen", "tipo"):
        if col in df.columns:
            out[f"revisados_por_{col}"] = {str(k): int(v) for k, v in df.loc[signed, col].value_counts().items()}
    return out


def criterion_row(df: pd.DataFrame) -> dict[str, Any]:
    """Fila del criterio del acta, en el mismo formato que `success_criteria_v3.json`."""
    s = review_summary(df)
    return {"criterio": ">= 20 errores revisados por el equipo",
            "valor": {"casos": s["casos"], "comentados": s["revisados"],
                      "comentados_sin_firma": s["comentados_sin_firma"], "revisores": s["revisores"]},
            "estado": "cumplido" if s["cumple"] else PENDING}


def carry_over_reviews(new: pd.DataFrame, previous_path: Path) -> pd.DataFrame:
    """Conserva comentarios y firmas ya cargados cuando se regenera el CSV (se cruzan por `id`).

    Sin esto, volver a correr `evaluate-v3` pisaría el trabajo de revisión del equipo.
    """
    if not previous_path.exists():
        return new
    try:
        old = load_errors(previous_path)
    except ReviewFileError:
        return new
    old = old[(old["comentario_equipo"].str.strip() != "") | (old["revisado_por"].str.strip() != "")]
    if old.empty:
        return new
    by_id = old.set_index("id")
    out = new.copy()
    for col in REVIEW_COLUMNS:
        kept = out["id"].astype(str).map(by_id[col])
        out[col] = kept.where(kept.notna(), out[col])
    return out


def update_review_status(reports: Path | None = None) -> dict[str, Any]:
    """Recalcula el criterio de revisión en `success_criteria_v3.json` sin tocar los otros criterios."""
    reports = reports or get_paths().reports
    df = load_errors(reports / "error_analysis_v3.csv")
    summary = review_summary(df)
    crit_path = reports / "success_criteria_v3.json"
    if crit_path.exists():
        crit = read_json(crit_path)
        row = criterion_row(df)
        hit = [i for i, c in enumerate(crit["criterios"]) if c["criterio"].startswith(CRITERION_PREFIX)]
        if not hit:
            raise ReviewFileError(f"{crit_path.name} no tiene el criterio de revisión de errores.")
        if crit["criterios"][hit[0]] != row:
            crit["criterios"][hit[0]] = row
            crit["revision_actualizada"] = dt.datetime.now().isoformat(timespec="seconds")
            write_json(crit, crit_path)
        summary["estado"] = row["estado"]
    return summary
