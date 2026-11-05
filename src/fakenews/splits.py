"""Preparación del dataset: limpieza, filtrado, deduplicación (antes de dividir) y splits."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .cleaning import build_document, word_count
from .config import get_paths, load_config, sha256_file, write_json
from .data import load_isot
from .dedup import connected_groups, cross_set_near_duplicates, exact_key, near_duplicate_pairs

LOGGER = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")


def _class_counts(df: pd.DataFrame) -> dict[str, int]:
    vc = df["label"].value_counts()
    return {"real": int(vc.get(0, 0)), "fake": int(vc.get(1, 0)), "total": int(len(df))}


def _drop_groups(df: pd.DataFrame, groups: list[int], step: str, log: list[dict[str, Any]]
                 ) -> pd.DataFrame:
    """Deja un representante por grupo; si el grupo mezcla etiquetas, lo descarta entero."""
    df = df.assign(_grp=groups)
    n_labels = df.groupby("_grp")["label"].transform("nunique")
    conflict = df[n_labels > 1]
    keep_mask = (n_labels == 1) & ~df.duplicated("_grp", keep="first")
    removed = df[~keep_mask]
    log.append({
        "paso": step,
        "filas_eliminadas": int(len(removed)),
        "eliminadas_por_clase": _class_counts(removed),
        "grupos_con_etiquetas_en_conflicto": int(conflict["_grp"].nunique()),
        "filas_en_conflicto": int(len(conflict)),
    })
    return df[keep_mask].drop(columns="_grp")


def prepare_dataset() -> dict[str, Any]:
    """Pipeline completo de preparación. Guarda splits en `data/processed/` y un manifiesto."""
    cfg, paths = load_config(), get_paths()
    seed = cfg["seed"]
    raw = load_isot()
    log: list[dict[str, Any]] = [{"paso": "carga", "filas": _class_counts(raw)}]

    LOGGER.info("Limpiando %d documentos (modo full y minimal)...", len(raw))
    raw["doc"] = [build_document(t, b, "full") for t, b in zip(raw["title"], raw["text"])]
    raw["doc_minimal"] = [build_document(t, b, "minimal") for t, b in zip(raw["title"], raw["text"])]
    raw["n_words"] = raw["doc"].map(word_count)

    min_words = cfg["data"]["min_words"]
    short = raw["n_words"] < min_words
    log.append({"paso": f"filtrar_textos_cortos (< {min_words} palabras tras limpiar)",
                "filas_eliminadas": int(short.sum()),
                "eliminadas_por_clase": _class_counts(raw[short])})
    df = raw[~short].reset_index(drop=True)

    keys = df["doc"].map(exact_key)
    key_groups = pd.factorize(keys)[0].tolist()
    df = _drop_groups(df, key_groups, "duplicados_exactos (texto limpio normalizado)", log)
    df = df.reset_index(drop=True)

    nd = cfg["data"]["near_duplicates"]
    LOGGER.info("Buscando casi-duplicados (MinHash+LSH) en %d documentos...", len(df))
    pairs = near_duplicate_pairs(df["doc"].tolist(), shingle_size=nd["shingle_size"],
                                 num_perm=nd["num_perm"], bands=nd["bands"],
                                 threshold=nd["jaccard_threshold"], seed=seed)
    df = _drop_groups(df, connected_groups(len(df), pairs),
                      f"casi_duplicados (Jaccard >= {nd['jaccard_threshold']} en shingles de "
                      f"{nd['shingle_size']} palabras)", log)
    log[-1]["pares_detectados"] = len(pairs)
    df = df.sort_values("id").reset_index(drop=True)
    log.append({"paso": "dataset_final", "filas": _class_counts(df)})

    sp = cfg["split"]
    train, rest = train_test_split(df, train_size=sp["train"], stratify=df["label"],
                                   random_state=seed)
    val, test = train_test_split(rest, test_size=sp["test"] / (sp["val"] + sp["test"]),
                                 stratify=rest["label"], random_state=seed)
    parts = {"train": train, "val": val, "test": test}

    cols = ["id", "label", "title", "subject", "date", "doc", "doc_minimal", "n_words"]
    manifest: dict[str, Any] = {"seed": seed, "pasos": log, "splits": {}}
    for name, part in parts.items():
        part = part.sort_values("id")[cols]
        out = paths.processed / f"{name}.csv"
        part.to_csv(out, index=False)
        ids_hash = hashlib.sha256("\n".join(part["id"]).encode()).hexdigest()
        manifest["splits"][name] = {**_class_counts(part), "sha256_archivo": sha256_file(out),
                                    "sha256_ids": ids_hash, "archivo": str(out.relative_to(paths.root))}
    manifest["solapamiento"] = check_overlap(parts)
    write_json(manifest, paths.reports / "splits_manifest.json")
    LOGGER.info("Splits guardados: %s", {k: v["total"] for k, v in manifest["splits"].items()})
    return manifest


def check_overlap(parts: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Verifica cero solapamiento entre splits: ids, duplicados exactos y casi-duplicados."""
    nd = load_config()["data"]["near_duplicates"]
    out: dict[str, Any] = {}
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ids = len(set(parts[a]["id"]) & set(parts[b]["id"]))
            ka = set(parts[a]["doc"].map(exact_key))
            kb = set(parts[b]["doc"].map(exact_key))
            near = cross_set_near_duplicates(parts[a]["doc"].tolist(), parts[b]["doc"].tolist(),
                                             shingle_size=nd["shingle_size"], num_perm=nd["num_perm"],
                                             bands=nd["bands"], threshold=nd["jaccard_threshold"],
                                             seed=load_config()["seed"])
            out[f"{a}-{b}"] = {"ids": ids, "exactos": len(ka & kb), "casi_duplicados": len(near)}
    out["ok"] = all(v["ids"] == 0 and v["exactos"] == 0 and v["casi_duplicados"] == 0
                    for k, v in out.items() if isinstance(v, dict))
    return out


def load_split(name: str, processed_dir: Path | None = None) -> pd.DataFrame:
    """Carga un split procesado (`train`, `val` o `test`)."""
    if name not in SPLITS:
        raise ValueError(f"Split desconocido: {name}. Opciones: {SPLITS}")
    path = (processed_dir or get_paths().processed) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Ejecutá `make data` (descarga + preparación).")
    df = pd.read_csv(path, dtype={"id": str, "title": str, "subject": str, "date": str,
                                  "doc": str, "doc_minimal": str}, keep_default_na=False)
    df["label"] = df["label"].astype(int)
    return df
