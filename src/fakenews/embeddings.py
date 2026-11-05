"""Experimento opcional con embeddings congelados (iteración 3, punto D).

El núcleo del proyecto NO depende de `sentence-transformers`. El cálculo se hace aparte
(`python -m fakenews.embeddings encode`, en un entorno con `requirements-extra.txt`) y deja una
caché en `data/interim/embeddings_v3.npz`. `EmbeddingLookup` solo la lee, por hash del texto.
"""

from __future__ import annotations

import hashlib
import sys
from functools import lru_cache
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .config import get_paths

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def text_key(text: str) -> bytes:
    return hashlib.sha1(text.encode("utf-8")).digest()[:12]  # noqa: S324 (clave de caché)


@lru_cache(maxsize=1)
def _store() -> tuple[dict[bytes, int], np.ndarray]:
    data = np.load(get_paths().interim / "embeddings_v3.npz")
    keys = data["keys"]
    return {k.tobytes(): i for i, k in enumerate(keys)}, data["vectors"]


class EmbeddingLookup(BaseEstimator, TransformerMixin):
    """Transformador: lista de textos -> vectores precalculados (falla si falta alguno)."""

    def fit(self, X: Any, y: Any = None) -> "EmbeddingLookup":  # noqa: N803
        return self

    def transform(self, X: Any) -> np.ndarray:  # noqa: N803
        index, vectors = _store()
        return vectors[[index[text_key(t)] for t in X]]


def collect_texts(profile: str = "v2_1") -> list[str]:
    """Todos los textos que necesita el panel (nunca test ni holdouts)."""
    from .config import load_config
    from .corpus import isot_split, liar_split, welfake_split
    from .panel import invariance_inputs_v3

    cfg = load_config()
    texts: list[str] = []
    for n in ("train", "val"):
        texts += isot_split(n, profile)["doc"].tolist()
    texts += welfake_split("dev", profile)["doc"].tolist() + liar_split("valid", profile)["doc"].tolist()
    inv = invariance_inputs_v3(profile, cfg["panel"]["invariance_sample"], cfg["seed"])
    for k, v in inv.items():
        if not k.startswith("__"):
            texts += v
    return sorted(set(texts))


def main() -> None:
    """`export <archivo>`: lista textos (entorno núcleo). `encode <archivo>`: calcula embeddings."""
    import json

    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == "export":
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(collect_texts(), fh, ensure_ascii=False)
        return
    from sentence_transformers import SentenceTransformer

    with open(path, encoding="utf-8") as fh:
        texts = json.load(fh)
    model = SentenceTransformer(MODEL)
    vecs = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    keys = np.array([np.frombuffer(text_key(t), dtype=np.uint8) for t in texts])
    np.savez_compressed(get_paths().interim / "embeddings_v3.npz", keys=keys, vectors=vecs.astype(np.float32))


if __name__ == "__main__":
    main()
