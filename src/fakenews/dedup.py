"""Detección de duplicados exactos y casi-duplicados (MinHash + LSH, implementado desde cero).

La idea es la misma similitud entre documentos que se ve en los labs 01/16, pero aplicada
a conjuntos de "shingles" (secuencias de 5 palabras): dos noticias son casi-duplicadas si su
similitud de Jaccard es alta. MinHash estima Jaccard con firmas cortas y LSH (bandas) evita
comparar todos contra todos. Los candidatos se verifican con Jaccard exacto.
"""

from __future__ import annotations

import hashlib
import re
import zlib
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_PRIME = np.uint64((1 << 61) - 1)
_MASK32 = np.uint64(0xFFFFFFFF)


def exact_key(text: str) -> str:
    """Clave para duplicados exactos: minúsculas, solo alfanuméricos, espacios colapsados."""
    norm = " ".join(_TOKEN.findall((text or "").lower()))
    return hashlib.md5(norm.encode("utf-8")).hexdigest()  # noqa: S324 (no es uso criptográfico)


def shingles(text: str, size: int = 5) -> set[int]:
    """Conjunto de shingles de `size` palabras, hasheados con CRC32 (determinista)."""
    toks = _TOKEN.findall((text or "").lower())
    if len(toks) < size:
        return {zlib.crc32(" ".join(toks).encode())} if toks else set()
    return {zlib.crc32(" ".join(toks[i:i + size]).encode()) for i in range(len(toks) - size + 1)}


def jaccard(a: set[int], b: set[int]) -> float:
    """Similitud de Jaccard entre dos conjuntos."""
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def minhash_signatures(shingle_sets: Sequence[set[int]], num_perm: int, seed: int) -> np.ndarray:
    """Firmas MinHash (n_docs x num_perm) con funciones hash (a*x + b) mod p."""
    rng = np.random.default_rng(seed)
    # a, b < 2^32 y x < 2^32 garantizan que a*x + b < 2^64 (sin desborde en uint64).
    a = rng.integers(1, 1 << 32, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, 1 << 32, size=num_perm, dtype=np.uint64)
    sigs = np.full((len(shingle_sets), num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, sh in enumerate(shingle_sets):
        if not sh:
            continue
        x = np.fromiter(sh, dtype=np.uint64, count=len(sh))[:, None]
        h = (a[None, :] * x + b[None, :]) % _PRIME
        sigs[i] = (h & _MASK32).min(axis=0)
    return sigs


def near_duplicate_pairs(texts: Sequence[str], shingle_size: int = 5, num_perm: int = 64,
                         bands: int = 16, threshold: float = 0.8, seed: int = 42
                         ) -> list[tuple[int, int, float]]:
    """Pares (i, j, jaccard) con Jaccard exacto >= threshold, encontrados vía LSH."""
    sets = [shingles(t, shingle_size) for t in texts]
    sigs = minhash_signatures(sets, num_perm, seed)
    rows = num_perm // bands
    candidates: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        block = sigs[:, band * rows:(band + 1) * rows]
        for i in range(len(texts)):
            if sets[i]:
                buckets[block[i].tobytes()].append(i)
        for members in buckets.values():
            if 1 < len(members) <= 500:
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        candidates.add((members[x], members[y]))
    out = []
    for i, j in sorted(candidates):
        s = jaccard(sets[i], sets[j])
        if s >= threshold:
            out.append((i, j, s))
    return out


def connected_groups(n: int, pairs: Sequence[tuple[int, int, float]]) -> list[int]:
    """Componentes conexas (union-find): devuelve el id de grupo de cada índice."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return [find(i) for i in range(n)]


def cross_set_near_duplicates(texts_a: Sequence[str], texts_b: Sequence[str], **kwargs
                              ) -> list[tuple[int, int, float]]:
    """Casi-duplicados entre dos colecciones (índices relativos a cada una)."""
    n_a = len(texts_a)
    pairs = near_duplicate_pairs(list(texts_a) + list(texts_b), **kwargs)
    return [(i, j - n_a, s) for i, j, s in pairs if i < n_a <= j]
