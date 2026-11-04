"""Representaciones del texto: TF-IDF (lab 16) y rasgos de estilo (acta, formularios 7 y 8)."""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .config import load_config

_WORD = re.compile(r"[A-Za-z][A-Za-z\-]*")
_TOKEN_LOWER = re.compile(r"(?u)\b\w\w+\b")

STYLE_FEATURES: list[str] = [
    "log_n_words",            # largo (log de palabras)
    "avg_word_len",           # largo medio de palabra
    "upper_char_ratio",       # proporción de letras mayúsculas
    "allcaps_word_ratio",     # proporción de palabras TODO MAYÚSCULAS (>= 2 letras)
    "exclam_per_100w",        # signos ! cada 100 palabras
    "question_per_100w",      # signos ? cada 100 palabras
    "sensational_ratio",      # proporción de palabras del léxico sensacionalista
    "attribution_ratio",      # proporción de verbos de atribución (said, told, according...)
    "first_person_ratio",     # I, we, our, us, me, my
    "second_person_ratio",    # you, your
    "repetition_ratio",       # 1 - palabras únicas / palabras totales
    "digit_ratio",            # proporción de caracteres numéricos
]
_FIRST = {"i", "we", "our", "us", "me", "my", "mine", "ours"}
_SECOND = {"you", "your", "yours"}


@lru_cache(maxsize=4)
def load_lexicon(name: str) -> frozenset[str]:
    """Carga un léxico de `resources/` (una palabra por línea, # = comentario)."""
    text = resources.files("fakenews.resources").joinpath(name).read_text(encoding="utf-8")
    return frozenset(w.strip().lower() for w in text.splitlines()
                     if w.strip() and not w.startswith("#"))


def style_vector(text: str) -> np.ndarray:
    """Calcula los rasgos de estilo de un texto (mismo orden que `STYLE_FEATURES`)."""
    return np.array(_style_tuple(text or ""), dtype=float)


@lru_cache(maxsize=200_000)
def _style_tuple(text: str) -> tuple[float, ...]:
    """Implementación cacheada (el stacking recalcula los rasgos en cada fold)."""
    words = _WORD.findall(text)
    n = len(words)
    if n == 0:
        return tuple(0.0 for _ in STYLE_FEATURES)
    lower = [w.lower() for w in words]
    letters = [c for c in text if c.isalpha()]
    sens, attr = load_lexicon("lexicon_sensational.txt"), load_lexicon("attribution_verbs.txt")
    return tuple(float(v) for v in (
        np.log1p(n),
        sum(len(w) for w in words) / n,
        sum(c.isupper() for c in letters) / max(1, len(letters)),
        sum(1 for w in words if len(w) >= 2 and w.isupper()) / n,
        100.0 * text.count("!") / n,
        100.0 * text.count("?") / n,
        sum(w in sens for w in lower) / n,
        sum(w in attr for w in lower) / n,
        sum(w in _FIRST for w in lower) / n,
        sum(w in _SECOND for w in lower) / n,
        1.0 - len(set(lower)) / n,
        sum(c.isdigit() for c in text) / max(1, len(text)),
    ))


class StyleFeatures(BaseEstimator, TransformerMixin):
    """Transformador sklearn: lista de textos -> matriz de rasgos de estilo."""

    def fit(self, X: Any, y: Any = None) -> "StyleFeatures":  # noqa: N803
        return self

    def transform(self, X: Any) -> np.ndarray:  # noqa: N803
        return np.vstack([style_vector(t) for t in X]) if len(X) else np.zeros((0, len(STYLE_FEATURES)))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(STYLE_FEATURES)


def make_tfidf(ngram_range: tuple[int, int] = (1, 2), max_features: int | None = None
               ) -> TfidfVectorizer:
    """Vectorizador TF-IDF con los parámetros de `config.yaml` (minúsculas, sublinear_tf)."""
    t = load_config()["tfidf"]
    return TfidfVectorizer(
        lowercase=t["lowercase"], sublinear_tf=t["sublinear_tf"], min_df=t["min_df"],
        max_df=t["max_df"], max_features=max_features or t["max_features"],
        ngram_range=tuple(ngram_range), strip_accents=t["strip_accents"], dtype=np.float32,
    )


def english_stopword_ratio(text: str) -> float:
    """Proporción de tokens que son stopwords del inglés (heurística simple de idioma)."""
    toks = [t.lower() for t in re.findall(r"[^\W\d_]+", text or "")]
    if not toks:
        return 0.0
    return sum(t in ENGLISH_STOP_WORDS for t in toks) / len(toks)


def non_ascii_letter_ratio(text: str) -> float:
    """Proporción de letras no ASCII (p. ej. ñ, á, cirílico): señal de otro idioma."""
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(ord(c) > 127 for c in letters) / len(letters)


def known_vocab_ratio(text: str, vocabulary: dict[str, int] | set[str]) -> float:
    """Proporción de tokens (unigramas, minúscula) presentes en el vocabulario del modelo."""
    toks = _TOKEN_LOWER.findall((text or "").lower())
    if not toks:
        return 0.0
    return sum(t in vocabulary for t in toks) / len(toks)
