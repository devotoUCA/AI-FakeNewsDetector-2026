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
