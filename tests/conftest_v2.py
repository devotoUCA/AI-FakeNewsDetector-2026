"""Marcadores de disponibilidad para los tests de la iteración 2."""

import pytest

from fakenews.config import get_paths


def _have(*rel: str) -> bool:
    root = get_paths().root
    return all((root / r).exists() for r in rel)


needs_v2_model = pytest.mark.skipif(
    not _have("models/final_model_v2.joblib", "models/retrieval_v2.joblib"),
    reason="Falta el modelo v2 (correr `python -m fakenews.cli train-v2` o `make quickstart`).")
needs_panel = pytest.mark.skipif(not _have("reports/panel_v2.json"), reason="Falta el panel v2.")
needs_welfake = pytest.mark.skipif(
    not _have("data/processed/welfake_dev.csv"), reason="Falta WELFake (prepare-external).")
