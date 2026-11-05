"""Corpus de la iteración 2: textos por perfil de limpieza (con caché), LIAR por porción y
WELFake como dataset externo de nivel artículo (desarrollo + holdout final).

Regla de evaluación: las decisiones usan ISOT train/val, LIAR *valid* y WELFake *dev*. El test
de ISOT y el holdout de WELFake solo se tocan en `evaluate_v2.evaluate_final_v2`.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .cleaning import build_document, clean_text, word_count
from .config import get_paths, load_config, sha256_file, write_json
from .data import DataValidationError, binarize_liar, load_isot, load_liar
from .dedup import connected_groups, cross_set_near_duplicates, exact_key, near_duplicate_pairs
from .splits import load_split

LOGGER = logging.getLogger(__name__)
WELFAKE_FILE = "WELFake_Dataset.csv"


# --- ISOT por perfil ---------------------------------------------------------------------
def _cache_path(name: str, profile: str) -> Path:
    return get_paths().interim / f"docs_{name}_{profile}.csv"


def isot_docs(profile: str) -> pd.Series:
    """Texto limpio de TODAS las filas finales de ISOT (train/val/test) con el perfil dado.

    Devuelve una Serie indexada por id. v1 sale de los splits procesados; v2/v3 se calculan
    desde el texto crudo y se cachean en `data/interim/`.
    """
    if profile == "v1":
        parts = [load_split(n)[["id", "doc"]] for n in ("train", "val", "test")]
        return pd.concat(parts).set_index("id")["doc"]
    cache = _cache_path("isot", profile)
    if cache.exists():
        df = pd.read_csv(cache, dtype=str, keep_default_na=False)
        return df.set_index("id")["doc"]
    ids = pd.concat([load_split(n)["id"] for n in ("train", "val", "test")])
    raw = load_isot().set_index("id").loc[ids]
    docs = [build_document(t, b, profile=profile) for t, b in zip(raw["title"], raw["text"])]
    out = pd.DataFrame({"id": ids.to_numpy(), "doc": docs})
    out.to_csv(cache, index=False)
    return out.set_index("id")["doc"]


def isot_split(name: str, profile: str) -> pd.DataFrame:
    """Un split de ISOT con la columna `doc` en el perfil pedido (y `doc_v1` como referencia)."""
    df = load_split(name)
    docs = isot_docs(profile)
    df = df.assign(doc_v1=df["doc"], doc=df["id"].map(docs).fillna(""))
    return df


def isot_raw(ids: pd.Series) -> pd.DataFrame:
    """Título y cuerpo crudos para una lista de ids (pruebas de invarianza)."""
    return load_isot().set_index("id").loc[ids, ["title", "text"]]


# --- LIAR ----------------------------------------------------------------------------------
def liar_split(split: str, profile: str) -> pd.DataFrame:
    """Porción oficial de LIAR (`train`, `valid` o `test`) binarizada, con `doc` limpio."""
    df = binarize_liar(load_liar())
    df = df[df["liar_split"] == split].copy()
    df["doc"] = [clean_text(s, profile=profile) for s in df["statement"]]
    df["label"] = df["y"].astype(int)
    df["n_words"] = df["doc"].map(word_count)
    return df.reset_index(drop=True)


# --- WELFake -------------------------------------------------------------------------------
def load_welfake_raw() -> pd.DataFrame:
    """CSV de WELFake (Zenodo 4561253, CC BY 4.0)."""
    path = get_paths().root / "data" / "raw" / "welfake" / WELFAKE_FILE
    if not path.exists():
        raise DataValidationError(
            f"Falta {path}. Ejecutá `python -m fakenews.cli download-external` o bajalo de "
            "https://zenodo.org/records/4561253 y dejalo en data/raw/welfake/.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"title", "text", "label"} - set(df.columns)
    if missing:
        raise DataValidationError(f"WELFake: faltan columnas {missing}")
    df = df.rename(columns={df.columns[0]: "row"})
    df["label"] = df["label"].astype(int)
    df["id"] = [f"wel_{int(r):05d}" for r in df["row"]]
    return df


def verify_welfake_label_direction(wel: pd.DataFrame, isot: pd.DataFrame) -> dict[str, Any]:
    """Verifica empíricamente qué significa cada etiqueta de WELFake usando las filas que
    coinciden textualmente con ISOT (cuya etiqueta conocemos). Falla si no es concluyente."""
    ik = dict(zip(isot["text"].map(exact_key), isot["label"]))
    k = wel["text"].map(exact_key)
    m = k.map(ik)
    sel = m.notna() & (wel["text"].str.strip() != "")
    agree = float((wel.loc[sel, "label"].to_numpy() == m[sel].astype(int).to_numpy()).mean())
    out = {"filas_coincidentes_con_isot": int(sel.sum()), "acuerdo_si_1_es_falsa": agree}
    if sel.sum() < 1000 or not (agree > 0.99 or agree < 0.01):
        raise DataValidationError(f"Dirección de etiquetas de WELFake no concluyente: {out}")
    out["direccion"] = "1 = falsa, 0 = real (igual que nuestra convención)" if agree > 0.99 else \
        "1 = real, 0 = falsa (se invierte)"
    out["invertir"] = bool(agree < 0.01)
    return out


def prepare_welfake() -> dict[str, Any]:
    """Prepara WELFake como externo: dirección de etiquetas verificada, limpieza v1 para
    filtrar/deduplicar, eliminación de todo lo que coincide (exacto o casi) con CUALQUIER fila
    de ISOT o con LIAR, y división estratificada 50/50 en `dev` y `holdout`."""
    cfg, paths = load_config(), get_paths()
    seed = cfg["seed"]
    nd = cfg["data"]["near_duplicates"]
    wel = load_welfake_raw()
    isot = load_isot()
    direction = verify_welfake_label_direction(wel, isot)
    if direction["invertir"]:
        wel["label"] = 1 - wel["label"]
    log: list[dict[str, Any]] = [{"paso": "carga", "filas": int(len(wel)),
                                  "falsas": int(wel["label"].sum())}]

    def step(name: str, keep: pd.Series) -> None:
        nonlocal wel
        removed = wel[~keep]
        log.append({"paso": name, "eliminadas": int(len(removed)),
                    "eliminadas_falsas": int(removed["label"].sum()),
                    "eliminadas_reales": int((removed["label"] == 0).sum())})
        wel = wel[keep].reset_index(drop=True)

    wel["doc"] = [build_document(t, b) for t, b in zip(wel["title"], wel["text"])]
    wel["n_words"] = wel["doc"].map(word_count)
    step(f"filtrar_textos_cortos (< {cfg['data']['min_words']} palabras)",
         wel["n_words"] >= cfg["data"]["min_words"])

    isot_docs_all = [build_document(t, b) for t, b in zip(isot["title"], isot["text"])]
    isot_keys = {exact_key(d) for d in isot_docs_all} | {exact_key(t) for t in isot["text"]}
    wel_keys = wel["doc"].map(exact_key)
    body_keys = wel["text"].map(exact_key)
    step("duplicado_exacto_de_cualquier_fila_de_ISOT",
         ~(wel_keys.isin(isot_keys) | body_keys.isin(isot_keys)))

    LOGGER.info("Casi-duplicados WELFake vs ISOT (MinHash)...")
    cross = cross_set_near_duplicates(wel["doc"].tolist(), isot_docs_all,
                                      shingle_size=nd["shingle_size"], num_perm=nd["num_perm"],
                                      bands=nd["bands"], threshold=nd["jaccard_threshold"], seed=seed)
    near_isot = np.zeros(len(wel), dtype=bool)
    near_isot[[i for i, _, _ in cross]] = True
    step("casi_duplicado_de_ISOT", pd.Series(~near_isot))

    liar_keys = set(load_liar()["statement"].map(exact_key))
    step("coincide_con_afirmacion_de_LIAR", ~wel["doc"].map(exact_key).isin(liar_keys))

    keys = wel["doc"].map(exact_key)
    grp = pd.factorize(keys)[0]
    nlab = pd.Series(wel["label"]).groupby(grp).transform("nunique").to_numpy()
    first = ~pd.Series(grp).duplicated().to_numpy()
    step("duplicados_exactos_internos (conflicto de etiqueta: se descarta el grupo)",
         pd.Series((nlab == 1) & first))

    pairs = near_duplicate_pairs(wel["doc"].tolist(), shingle_size=nd["shingle_size"],
                                 num_perm=nd["num_perm"], bands=nd["bands"],
                                 threshold=nd["jaccard_threshold"], seed=seed)
    grp = np.array(connected_groups(len(wel), pairs))
    nlab = pd.Series(wel["label"]).groupby(grp).transform("nunique").to_numpy()
    first = ~pd.Series(grp).duplicated().to_numpy()
    step("casi_duplicados_internos", pd.Series((nlab == 1) & first))

    reuters_left = float(wel["text"].str.contains(r"\(Reuters\)").mean())
    dev, hold = train_test_split(wel, test_size=0.5, stratify=wel["label"], random_state=seed)
    out: dict[str, Any] = {"fuente": "WELFake, Zenodo record 4561253 (CC BY 4.0)",
                           "sha256_csv": sha256_file(paths.root / "data/raw/welfake" / WELFAKE_FILE),
                           "direccion_etiquetas": direction, "pasos": log,
                           "frac_con_dateline_reuters_tras_dedup": reuters_left, "splits": {}}
    for name, part in (("dev", dev), ("holdout", hold)):
        part = part.sort_values("id")[["id", "label", "title", "text", "doc", "n_words"]]
        path = paths.processed / f"welfake_{name}.csv"
        part.to_csv(path, index=False)
        out["splits"][name] = {"n": int(len(part)), "falsas": int(part["label"].sum()),
                               "reales": int((part["label"] == 0).sum()),
                               "sha256_ids": hashlib.sha256("\n".join(part["id"]).encode()).hexdigest()}
    write_json(out, paths.reports / "welfake_manifest_v2.json")
    return out


def welfake_split(name: str, profile: str) -> pd.DataFrame:
    """`dev` o `holdout` de WELFake con `doc` en el perfil pedido (v2/v3 cacheados)."""
    if name not in ("dev", "holdout"):
        raise ValueError("WELFake: split debe ser 'dev' o 'holdout'")
    path = get_paths().processed / f"welfake_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Ejecutá `python -m fakenews.cli prepare-external`.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["label"] = df["label"].astype(int)
    if profile != "v1":
        cache = _cache_path(f"welfake_{name}", profile)
        if cache.exists():
            c = pd.read_csv(cache, dtype=str, keep_default_na=False).set_index("id")["doc"]
            df["doc"] = df["id"].map(c).fillna("")
        else:
            df["doc"] = [build_document(t, b, profile=profile) for t, b in zip(df["title"], df["text"])]
            df[["id", "doc"]].to_csv(cache, index=False)
    return df
