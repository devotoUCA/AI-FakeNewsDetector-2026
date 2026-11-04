"""Artefactos: recarga reproducible, metadata completa y regresión de fuga."""

import joblib
import numpy as np

from fakenews.config import get_paths, read_json
from fakenews.evaluate import SOURCE_TOKENS
from fakenews.explain import global_top_terms
from fakenews.models import p_fake
from fakenews.splits import load_split

from .conftest import needs_data, needs_model


@needs_model
@needs_data
def test_saved_model_reproduces_predictions():
    paths = get_paths()
    a = joblib.load(paths.models / "final_model.joblib")
    b = joblib.load(paths.models / "final_model.joblib")
    val = load_split("val").head(300)
    pa, pb = p_fake(a["model"], val["doc"].tolist()), p_fake(b["model"], val["doc"].tolist())
    assert np.array_equal(pa, pb)
    # y coinciden con las predicciones guardadas al entrenar
    saved = np.load(paths.interim / "val_predictions.npz")
    full_val = load_split("val")
    assert np.allclose(p_fake(a["model"], full_val["doc"].head(300).tolist()),
                       saved[a["name"]][:300], atol=1e-6)


@needs_model
def test_metadata_present():
    meta = read_json(get_paths().models / "metadata.json")
    for key in ("modelo", "version", "semilla", "config_hash", "umbral", "datos", "versiones", "creado"):
        assert key in meta
    assert meta["semilla"] == 42
    assert {"train", "val", "test"} <= set(meta["datos"])
    assert all(len(v["sha256_ids"]) == 64 for v in meta["datos"].values())
    assert {"python", "sklearn", "numpy", "pandas"} <= set(meta["versiones"])


@needs_model
def test_no_source_tokens_among_top_terms():
    """Regresión de fuga: ningún token de fuente/plataforma entre los 50 más influyentes."""
    art = joblib.load(get_paths().models / "final_model.joblib")
    top = global_top_terms(art["model"], 50)
    terms = {d["termino"] for d in top["hacia_falsa"] + top["hacia_real"]}
    leaked = terms & SOURCE_TOKENS
    assert not leaked, f"Tokens de fuente entre los más influyentes: {sorted(leaked)}"
