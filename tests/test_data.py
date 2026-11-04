"""Tests de carga, validación, mapeo de etiquetas y splits."""

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from fakenews.config import get_paths, load_config, read_json, sha256_file
from fakenews.data import DataValidationError, binarize_liar, load_isot, validate_frame
from fakenews.dedup import exact_key
from fakenews.splits import load_split

from .conftest import needs_data


def test_validate_frame_errors():
    with pytest.raises(DataValidationError, match="faltan columnas"):
        validate_frame(pd.DataFrame({"title": ["x"]}), ["title", "text"], "x.csv")
    with pytest.raises(DataValidationError, match="vacío"):
        validate_frame(pd.DataFrame({"title": []}), ["title"], "x.csv")


def test_load_isot_from_tmp(tmp_path):
    # FIXTURE SINTÉTICO mínimo para verificar columnas, tipos y mapeo de etiquetas.
    row = {"title": "t", "text": "body", "subject": "s", "date": "May 1, 2017"}
    pd.DataFrame([row, row]).to_csv(tmp_path / "Fake.csv", index=False)
    pd.DataFrame([row]).to_csv(tmp_path / "True.csv", index=False)
    df = load_isot(tmp_path)
    assert list(df.columns) == ["id", "label", "title", "text", "subject", "date", "source_file"]
    assert df["label"].tolist() == [1, 1, 0]  # Falsa = 1 (positiva), Real = 0
    assert df["id"].is_unique and all(isinstance(x, str) for x in df["text"])


def test_load_isot_missing_file_message(tmp_path):
    with pytest.raises(DataValidationError, match="make data"):
        load_isot(tmp_path)


def test_binarize_liar():
    df = pd.DataFrame({"label": ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]})
    out = binarize_liar(df)
    assert out["label"].tolist() == ["true", "mostly-true", "false", "pants-fire"]
    assert out["y"].tolist() == [0, 0, 1, 1]


@needs_data
def test_splits_no_overlap_and_no_duplicates():
    parts = {n: load_split(n) for n in ("train", "val", "test")}
    ids = [set(p["id"]) for p in parts.values()]
    keys = [set(p["doc"].map(exact_key)) for p in parts.values()]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not ids[i] & ids[j]
            assert not keys[i] & keys[j]
    allk = pd.concat([p["doc"] for p in parts.values()]).map(exact_key)
    assert allk.is_unique
    manifest = read_json(get_paths().reports / "splits_manifest.json")
    assert manifest["solapamiento"]["ok"]  # incluye casi-duplicados (MinHash) entre splits


@needs_data
def test_split_files_match_manifest_hash():
    manifest = read_json(get_paths().reports / "splits_manifest.json")
    for name, info in manifest["splits"].items():
        assert sha256_file(get_paths().root / info["archivo"]) == info["sha256_archivo"]


@needs_data
def test_stratification_and_seed_reproducibility():
    cfg = load_config()
    parts = {n: load_split(n) for n in ("train", "val", "test")}
    rates = [p["label"].mean() for p in parts.values()]
    assert max(rates) - min(rates) < 0.005  # estratificado
    sizes = {n: len(p) for n, p in parts.items()}
    total = sum(sizes.values())
    assert abs(sizes["train"] / total - cfg["split"]["train"]) < 0.002
    # Mismo seed -> mismos splits: se rehace la división desde el dataset final.
    df = pd.concat(parts.values()).sort_values("id").reset_index(drop=True)
    tr, rest = train_test_split(df, train_size=cfg["split"]["train"], stratify=df["label"],
                                random_state=cfg["seed"])
    va, te = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=cfg["seed"])
    assert set(tr["id"]) == set(parts["train"]["id"])
    assert set(te["id"]) == set(parts["test"]["id"])


@needs_data
def test_processed_docs_have_no_source_marks():
    from fakenews.audit import residual_marker_rates

    for name in ("train", "test"):
        rates = residual_marker_rates(load_split(name)["doc"])
        for marker in ("dateline '(Reuters) -'", "palabra 'Reuters'", "pic.twitter.com", "URL (http/www)",
                       "'Getty Images'", "descargo SAP"):
            assert rates[marker] == 0.0, (name, marker)
