"""Descarga, carga y validación de los datasets (ISOT/Kaggle como principal, LIAR como externo)."""

from __future__ import annotations

import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from .config import get_paths, load_config, sha256_file

LOGGER = logging.getLogger(__name__)

ISOT_COLUMNS = ["title", "text", "subject", "date"]
LIAR_COLUMNS = [
    "id", "label", "statement", "subjects", "speaker", "speaker_job", "state", "party",
    "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts",
    "pants_on_fire_counts", "context",
]
LIAR_LABELS = {"true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"}


class DataValidationError(ValueError):
    """Error de validación de datos con un mensaje accionable."""


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Descargando %s -> %s", url, dest)
    req = urllib.request.Request(url, headers={"User-Agent": "fakenews-uca-course-project"})
    with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as fh:  # noqa: S310
        while chunk := resp.read(1 << 20):
            fh.write(chunk)


def download_datasets(force: bool = False) -> dict[str, Any]:
    """Descarga y extrae ISOT (Kaggle, endpoint público) y LIAR; verifica SHA-256.

    No borra ni sobrescribe archivos existentes salvo ``force=True``.
    """
    cfg, paths = load_config(), get_paths()
    out: dict[str, Any] = {}
    for key, target in (("isot", paths.raw_isot), ("liar", paths.raw_liar)):
        spec = cfg["download"][key]
        expected: dict[str, str] = spec["sha256"]
        if not force and all((target / f).exists() for f in expected):
            LOGGER.info("%s ya está en %s (no se descarga)", key, target)
        else:
            zpath = target / spec["zip_name"]
            try:
                if force or not zpath.exists():
                    _download(spec["url"], zpath)
                with zipfile.ZipFile(zpath) as zf:
                    for member in zf.namelist():
                        name = Path(member).name
                        if name in expected:  # se extrae solo lo esperado, sin rutas
                            (target / name).write_bytes(zf.read(member))
            except Exception as exc:  # noqa: BLE001
                raise DataValidationError(
                    f"No se pudo descargar {key} desde {spec['url']} ({exc}). Descargalo a mano y "
                    f"dejá {sorted(expected)} en {target}/"
                ) from exc
        hashes = {f: sha256_file(target / f) for f in expected}
        bad = {f: h for f, h in hashes.items() if h != expected[f]}
        if bad:
            LOGGER.warning("SHA-256 distinto al registrado para %s: %s (¿nueva versión?)", key, bad)
        out[key] = {"dir": str(target), "sha256": hashes, "hash_ok": not bad}
    return out


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise DataValidationError(f"Falta {path}. {hint}")


def validate_frame(df: pd.DataFrame, columns: list[str], name: str) -> None:
    """Verifica columnas esperadas y tipo texto; falla con un mensaje claro."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"{name}: faltan columnas {missing}. Columnas encontradas: {list(df.columns)}"
        )
    if len(df) == 0:
        raise DataValidationError(f"{name}: el archivo está vacío.")


def load_isot(raw_dir: Path | None = None) -> pd.DataFrame:
    """Carga `Fake.csv` y `True.csv` con un id estable por fila y la etiqueta (Falsa=1, Real=0).

    Columnas devueltas: id, label, title, text, subject, date, source_file.
    """
    cfg = load_config()
    raw_dir = raw_dir or get_paths().raw_isot
    frames = []
    for fname, label, prefix in (("Fake.csv", cfg["labels"]["fake"], "fake"),
                                 ("True.csv", cfg["labels"]["real"], "true")):
        path = raw_dir / fname
        _require(path, "Ejecutá `make data` (o `python -m fakenews.cli download`).")
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        validate_frame(df, ISOT_COLUMNS, fname)
        df = df[ISOT_COLUMNS].copy()
        df.insert(0, "id", [f"{prefix}_{i:05d}" for i in range(len(df))])
        df.insert(1, "label", int(label))
        df["source_file"] = fname
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for c in ISOT_COLUMNS:
        out[c] = out[c].astype(str)
    if not set(out["label"].unique()) <= {0, 1}:
        raise DataValidationError("Etiquetas fuera de {0, 1} tras el mapeo.")
    return out


def load_liar(raw_dir: Path | None = None) -> pd.DataFrame:
    """Carga los tres TSV de LIAR (sin encabezado) y agrega la columna `liar_split`."""
    raw_dir = raw_dir or get_paths().raw_liar
    frames = []
    for split in ("train", "valid", "test"):
        path = raw_dir / f"{split}.tsv"
        _require(path, "Ejecutá `make data` para descargar LIAR.")
        df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS, dtype=str,
                         keep_default_na=False, quoting=3)
        validate_frame(df, LIAR_COLUMNS, f"LIAR {split}")
        df["liar_split"] = split
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    unknown = set(out["label"]) - LIAR_LABELS
    if unknown:
        raise DataValidationError(f"LIAR: etiquetas inesperadas {unknown}")
    return out


def binarize_liar(df: pd.DataFrame) -> pd.DataFrame:
    """Binariza LIAR según `config.yaml`: falsa=1, real=0; excluye las etiquetas intermedias."""
    ext = load_config()["external"]
    fake, real = set(map(str, ext["liar_fake_labels"])), set(map(str, ext["liar_real_labels"]))
    # YAML lee `false`/`true` como booleanos: se normalizan a texto en minúscula.
    fake = {s.lower() for s in fake}
    real = {s.lower() for s in real}
    keep = df[df["label"].isin(fake | real)].copy()
    keep["y"] = keep["label"].isin(fake).astype(int)
    return keep
