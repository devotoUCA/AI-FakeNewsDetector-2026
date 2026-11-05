"""Pruebas de invarianza (metamórficas): transformaciones que NO cambian la veracidad.

Se aplican sobre el texto CRUDO (título y cuerpo) y luego se limpia con el perfil del modelo,
igual que en inferencia. Un modelo que usa atajos de formato cambia de opinión; uno robusto no.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

_CURLY = re.compile(r"[’‘“”]")
_CONTRACTION_APOS = re.compile(r"\b(\w+)[’'](t|s|re|ll|ve|d|m)\b")
_CONTRACTION_SPACE = re.compile(r"\b(\w+n|it|that|he|she|there|what|[A-Z][a-z]+) (t|s)\b")
_DATELINE = re.compile(r"^\s*(?:[A-Z][A-Z\.\-'’]+(?:[ /][A-Z][A-Z\.\-'’]+)*(?:,\s*[A-Z][A-Za-z\.]{1,20}){0,2}\s*)?"
                       r"\(Reuters\)\s*[-–—]*\s*")
_DAYS_MONTHS = re.compile(r"\b(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)days?\b|\b(?:January|February|March|"
                          r"April|June|July|August|September|October|November|December)\b")


def swap_quote_style(title: str, body: str) -> tuple[str, str]:
    """Comillas curvas <-> rectas."""
    def f(s: str) -> str:
        if _CURLY.search(s):
            return s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        out, open_q = [], True
        for ch in s:
            if ch == '"':
                out.append("“" if open_q else "”")
                open_q = not open_q
            else:
                out.append("’" if ch == "'" else ch)
        return "".join(out)
    return f(title), f(body)


def swap_apostrophe_space(title: str, body: str) -> tuple[str, str]:
    """Apóstrofo de contracción <-> espacio ("don't" <-> "don t")."""
    def f(s: str) -> str:
        if _CONTRACTION_APOS.search(s):
            return _CONTRACTION_APOS.sub(r"\1 \2", s)
        return _CONTRACTION_SPACE.sub(r"\1'\2", s)
    return f(title), f(body)


def swap_title_case(title: str, body: str) -> tuple[str, str]:
    """Título en MAYÚSCULAS <-> título en minúsculas con mayúscula inicial."""
    if re.search(r"\b[A-Z]{4,}\b", title):
        return title[:1].upper() + title[1:].lower(), body
    return title.upper(), body


def drop_days_months(title: str, body: str) -> tuple[str, str]:
    """Quita nombres de días y meses."""
    return _DAYS_MONTHS.sub("", title), _DAYS_MONTHS.sub("", body)


def swap_source_marks(title: str, body: str) -> tuple[str, str]:
    """Quita la marca de agencia si está; si no está, la agrega."""
    if "(Reuters)" in body:
        return title, re.sub(r"\bReuters\b", "", _DATELINE.sub("", body))
    return title, "WASHINGTON (Reuters) - " + body


TRANSFORMS: dict[str, Callable[[str, str], tuple[str, str]]] = {
    "comillas_curvas_rectas": swap_quote_style,
    "apostrofo_espacio": swap_apostrophe_space,
    "mayusculas_titulo": swap_title_case,
    "sin_dias_meses": drop_days_months,
    "marcas_de_fuente": swap_source_marks,
}


def flip_stats(p_base: np.ndarray, p_new: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Tasa de cambio de predicción y cambio medio de probabilidad."""
    return {"tasa_cambio": float(np.mean((p_base >= threshold) != (p_new >= threshold))),
            "delta_p_medio": float(np.mean(np.abs(p_new - p_base)))}


# --- Iteración 3: transformaciones de formato y tasa de cambio con intervalo de confianza --------
def swap_sentence_glue(title: str, body: str) -> tuple[str, str]:
    """Oración pegada <-> separada ("it.The" <-> "it. The")."""
    if re.search(r"[a-z]\.[A-Z]", body):
        return title, re.sub(r"([a-z])\.([A-Z])", r"\1. \2", body)
    return title, re.sub(r"([a-z])\. ([A-Z])", r"\1.\2", body)


def swap_double_spaces(title: str, body: str) -> tuple[str, str]:
    """Doble espacio <-> espacio simple tras punto."""
    if "  " in body:
        return title, re.sub(r" {2,}", " ", body)
    return title, body.replace(". ", ".  ")


def swap_dashes(title: str, body: str) -> tuple[str, str]:
    """Guion largo <-> guion corto."""
    if "—" in body or "–" in body:
        return title, body.replace("—", "-").replace("–", "-")
    return title, body.replace(" - ", " — ")


def swap_ellipsis(title: str, body: str) -> tuple[str, str]:
    """Puntos suspensivos Unicode <-> ASCII."""
    if "…" in body:
        return title, body.replace("…", "...")
    return title, body.replace("...", "…")


def space_before_punct(title: str, body: str) -> tuple[str, str]:
    """Agrega un espacio antes de comas y puntos (hueco típico de texto editado)."""
    return title, re.sub(r"(?<=[a-z])([,.])(?=\s)", r" \1", body)


def hard_spaces(title: str, body: str) -> tuple[str, str]:
    """Espacio común -> espacio duro después de puntuación."""
    return title, re.sub(r"([,.;:]) ", "\\1\u00a0", body)


TRANSFORMS_V3: dict[str, Callable[[str, str], tuple[str, str]]] = {
    **TRANSFORMS,
    "oraciones_pegadas": swap_sentence_glue,
    "espacios_dobles": swap_double_spaces,
    "guiones": swap_dashes,
    "puntos_suspensivos": swap_ellipsis,
    "espacio_antes_de_puntuacion": space_before_punct,
    "espacio_duro": hard_spaces,
}


def flip_stats_ci(p_base: np.ndarray, p_new: np.ndarray, threshold: float, seed: int,
                  n_resamples: int = 2000) -> dict[str, float]:
    """Tasa de cambio con intervalo de confianza 95 % por bootstrap percentil."""
    flips = ((p_base >= threshold) != (p_new >= threshold)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(flips), (n_resamples, len(flips)))
    means = flips[idx].mean(axis=1)
    return {"tasa_cambio": float(flips.mean()), "ic95_inf": float(np.percentile(means, 2.5)),
            "ic95_sup": float(np.percentile(means, 97.5)), "n": int(len(flips)),
            "delta_p_medio": float(np.mean(np.abs(p_new - p_base)))}
