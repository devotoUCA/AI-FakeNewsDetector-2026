"""Marcadores de disponibilidad para los tests de la iteración 3."""

import pytest

from fakenews.config import get_paths


def _have(*rel: str) -> bool:
    root = get_paths().root
    return all((root / r).exists() for r in rel)


needs_v3_model = pytest.mark.skipif(
    not _have("models/final_model_v3.joblib", "models/retrieval_v3.joblib"),
    reason="Falta el modelo v3 (correr `python -m fakenews.tasks quickstart`).")
needs_v3_reports = pytest.mark.skipif(
    not _have("reports/final_v3.json", "reports/invariance_v3.json", "reports/shrink_resolution_v3.json"),
    reason="Faltan los reportes de la iteración 3.")
