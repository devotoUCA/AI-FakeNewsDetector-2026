"""Carga de la configuración única (`config.yaml`) y utilidades comunes."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOGGER = logging.getLogger("fakenews")


def project_root() -> Path:
    """Devuelve la raíz del repo (directorio que contiene `config.yaml`).

    Se puede forzar con la variable de entorno `FAKENEWS_ROOT`.
    """
    env = os.environ.get("FAKENEWS_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "config.yaml").exists():
            return parent
    return Path.cwd()


@dataclass(frozen=True)
class Paths:
    """Rutas absolutas derivadas de la configuración (relativas a la raíz del repo)."""

    root: Path
    raw_isot: Path
    raw_liar: Path
    interim: Path
    processed: Path
    models: Path
    reports: Path
    figures: Path


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Lee `config.yaml` y devuelve un diccionario (cacheado)."""
    cfg_path = Path(path) if path else project_root() / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No se encontró la configuración en {cfg_path}. Ejecutá desde la raíz del repo."
        )
    with cfg_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_paths(cfg: dict[str, Any] | None = None) -> Paths:
    """Construye las rutas del proyecto y crea los directorios de salida si faltan."""
    cfg = cfg or load_config()
    root = project_root()
    p = {k: (root / v).resolve() for k, v in cfg["paths"].items()}
    paths = Paths(root=root, **p)
    for d in (paths.interim, paths.processed, paths.models, paths.reports, paths.figures):
        d.mkdir(parents=True, exist_ok=True)
    return paths


def set_seed(seed: int | None = None) -> int:
    """Fija la semilla global (random y numpy) a partir de la configuración."""
    seed = load_config()["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    return seed


def config_hash() -> str:
    """Hash corto de la configuración, para trazabilidad de artefactos y reportes."""
    blob = json.dumps(load_config(), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    """SHA-256 de un archivo."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(obj: Any, path: Path) -> None:
    """Guarda un objeto como JSON legible (UTF-8, sin escapar acentos)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=_json_default)
        fh.write("\n")


def read_json(path: Path) -> Any:
    """Lee un JSON."""
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"No serializable: {type(o)}")


def setup_logging(level: int = logging.INFO) -> None:
    """Configura el logging del paquete (formato simple, en consola)."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
