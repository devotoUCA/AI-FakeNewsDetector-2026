"""Auditoría de calidad y de fuga de información (lab 03: faltantes, atípicos, fuga).

Todo se mide empíricamente sobre los datos reales; los resultados van a
`reports/data_audit.json` y `reports/leakage_audit.json`.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from .cleaning import build_document, word_count
from .config import get_paths, load_config, write_json
from .data import load_isot
from .splits import load_split

LOGGER = logging.getLogger(__name__)

# Marcas candidatas a fuga (en el texto crudo). Se mide qué fracción de cada clase las tiene.
TEXT_MARKERS: dict[str, str] = {
    "dateline '(Reuters) -'": r"\(Reuters\)\s*[-–—]",
    "palabra 'Reuters'": r"\bReuters\b",
    "'Featured image'": r"(?i)featured image",
    "'Getty Images'": r"(?i)getty images",
    "pic.twitter.com": r"pic\.twitter\.com",
    "URL (http/www)": r"(?i)https?://|\bwww\.",
    "@usuario": r"@\w+",
    "firma de tuit '(@x) Mes D, AAAA'": r"\(@\w+\)\s+[A-Z][a-z]+\.? \d{1,2}, \d{4}",
    "'Via:' / 'Read more:'": r"(?i)\b(?:via|read more)\s*:",
    "21st Century Wire / ACR": r"(?i)21st century wire|21wire|alternate current radio",
    "descargo SAP": r"funded in part by SAP|SAP had no editorial",
    "comilla/apóstrofo curvo (’ “ ”)": r"[’“”]",
    "apóstrofo convertido en espacio ('don t')": r"\b(?:don|doesn|isn|wasn|can|won|didn|it|that)\s[ts]\b",
    "espacio duro (\\xa0)": "\xa0",
}
TITLE_MARKERS: dict[str, str] = {
    "título con (VIDEO)/[VIDEO]": r"[\(\[]\s*VIDEO",
    "título con palabra en MAYÚSCULAS (>=4 letras)": r"\b[A-Z]{4,}\b",
    "título con '!'": r"!",
    "título 'Factbox:'/'Exclusive:'": r"^(?:Factbox|Exclusive)\s*:",
}


def _rate(series: pd.Series, pattern: str) -> float:
    return float(series.str.contains(pattern, regex=True).mean())


def marker_table(df: pd.DataFrame, text_col: str = "text", title_col: str = "title"
                 ) -> list[dict[str, Any]]:
    """Fracción de cada clase que contiene cada marca, y accuracy de la regla 'marca => clase'."""
    rows = []
    y = df["label"].to_numpy()
    for markers, col in ((TEXT_MARKERS, text_col), (TITLE_MARKERS, title_col)):
        for name, pat in markers.items():
            has = df[col].str.contains(pat, regex=True).to_numpy()
            fake_rate = float(has[y == 1].mean())
            real_rate = float(has[y == 0].mean())
            # Regla de una sola marca: si la marca es más frecuente en falsas, predice falsa.
            pred = has.astype(int) if fake_rate >= real_rate else (~has).astype(int)
            rows.append({"marca": name, "columna": col, "frac_falsas": round(fake_rate, 4),
                         "frac_reales": round(real_rate, 4),
                         "accuracy_regla_sola": round(float(accuracy_score(y, pred)), 4)})
    return rows


def _metadata_only_models(train: pd.DataFrame, val: pd.DataFrame) -> dict[str, Any]:
    """¿Cuánto predicen `subject` y `date` por sí solas? (entrena en train, mide en val)."""
    out: dict[str, Any] = {}
    seed = load_config()["seed"]
    vec = CountVectorizer(token_pattern=r".+", lowercase=False)
    xs_tr, xs_va = vec.fit_transform(train["subject"]), vec.transform(val["subject"])
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(xs_tr, train["label"])
    out["solo_subject"] = {
        "accuracy_val": round(float(accuracy_score(val["label"], clf.predict(xs_va))), 4),
        "subjects_falsas": sorted(train.loc[train.label == 1, "subject"].unique().tolist()),
        "subjects_reales": sorted(train.loc[train.label == 0, "subject"].unique().tolist()),
    }

    def date_feats(d: pd.DataFrame) -> np.ndarray:
        dt = pd.to_datetime(d["date"].str.strip(), errors="coerce", format="mixed")
        return np.c_[dt.dt.year.fillna(0), dt.dt.month.fillna(0), dt.dt.day.fillna(0),
                     dt.isna().astype(int)]

    tree = DecisionTreeClassifier(max_depth=6, random_state=seed)
    tree.fit(date_feats(train), train["label"])
    out["solo_fecha (año, mes, día)"] = {
        "accuracy_val": round(float(accuracy_score(val["label"], tree.predict(date_feats(val)))), 4)}

    def fmt_feats(d: pd.DataFrame) -> np.ndarray:
        s = d["date"]
        return np.c_[s.str.endswith(" ").astype(int),
                     s.str.match(r"^[A-Z][a-z]{2}\s").astype(int),
                     s.str.match(r"^[A-Z][a-z]+ \d").astype(int)]

    tree2 = DecisionTreeClassifier(max_depth=3, random_state=seed)
    tree2.fit(fmt_feats(train), train["label"])
    out["solo_formato_de_fecha (espacio final, mes abreviado)"] = {
        "accuracy_val": round(float(accuracy_score(val["label"], tree2.predict(fmt_feats(val)))), 4)}
    return out


def data_quality(raw: pd.DataFrame) -> dict[str, Any]:
    """Faltantes, vacíos, atípicos y duplicados del dataset crudo (por clase)."""
    out: dict[str, Any] = {}
    raw = raw.assign(n_chars=raw["text"].str.len(), n_words_raw=raw["text"].map(word_count))
    for lab, name in ((1, "falsas"), (0, "reales")):
        d = raw[raw.label == lab]
        dates = pd.to_datetime(d["date"].str.strip(), errors="coerce", format="mixed")
        out[name] = {
            "filas": int(len(d)),
            "nulos_por_columna": {c: int((d[c].str.strip() == "").sum()) for c in
                                  ("title", "text", "subject", "date")},
            "texto_vacio_o_solo_espacios": int((d["text"].str.strip() == "").sum()),
            "texto_menos_de_20_palabras": int((d["n_words_raw"] < 20).sum()),
            "texto_solo_enlaces": int(d["text"].str.fullmatch(r"\s*(?:https?://\S+\s*)+").sum()),
            "fechas_no_parseables": int(dates.isna().sum()),
            "ejemplos_fechas_no_parseables": d.loc[dates.isna(), "date"].head(3).tolist(),
            "rango_fechas": [str(dates.min().date()), str(dates.max().date())],
            "palabras_cuerpo": {k: round(float(v), 1) for k, v in
                                d["n_words_raw"].describe(percentiles=[.01, .5, .99]).items()},
            "textos_mas_de_5000_palabras": int((d["n_words_raw"] > 5000).sum()),
            "duplicados_exactos_texto": int(d["text"].duplicated().sum()),
            "duplicados_exactos_titulo_y_texto": int(d.duplicated(["title", "text"]).sum()),
            "subjects": d["subject"].value_counts().to_dict(),
        }
    out["textos_identicos_entre_clases"] = int(len(
        set(raw.loc[raw.label == 1, "text"].str.strip()) & set(raw.loc[raw.label == 0, "text"].str.strip())
        - {""}))
    return out


def run_audit() -> dict[str, Any]:
    """Corre la auditoría completa (requiere los splits ya preparados)."""
    paths = get_paths()
    raw = load_isot()
    LOGGER.info("Auditoría de calidad de datos...")
    quality = data_quality(raw)
    write_json(quality, paths.reports / "data_audit.json")

    LOGGER.info("Auditoría de fuga (marcas en texto crudo y tras limpiar)...")
    before = marker_table(raw)
    clean = raw.assign(doc=[build_document(t, b) for t, b in zip(raw["title"], raw["text"])])
    clean["first_line"] = clean["doc"].str.split("\n").str[0]
    after = marker_table(clean, text_col="doc", title_col="first_line")
    train, val = load_split("train"), load_split("val")
    leakage = {
        "nota": "Accuracy de una regla que usa SOLO la presencia de la marca (sobre todo el dataset "
                "crudo). Valores cercanos a 1 indican que la marca separa las clases sin leer el "
                "contenido.",
        "marcas_texto_crudo": before,
        "marcas_tras_limpieza_full": after,
        "metadatos_solos": _metadata_only_models(train, val),
    }
    write_json(leakage, paths.reports / "leakage_audit.json")
    return {"calidad": quality, "fuga": leakage}


def residual_marker_rates(texts: pd.Series) -> dict[str, float]:
    """Fracción de textos con marcas residuales (para tests de regresión de limpieza)."""
    return {name: float(texts.str.contains(p, regex=True).mean()) for name, p in TEXT_MARKERS.items()
            if not re.search(r"apóstrofo convertido", name)}
