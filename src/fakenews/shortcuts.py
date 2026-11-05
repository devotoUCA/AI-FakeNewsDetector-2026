"""Auditoría de atajos de un solo rasgo (iteración 2, punto A).

Para cada rasgo superficial se mide cuánto predice la etiqueta POR SÍ SOLO: AUC orientada
(max(AUC, 1-AUC)) y accuracy de un "stump" (árbol de profundidad 1) entrenado en ISOT train y
medido en ISOT val. Se mide sobre el texto crudo y sobre cada perfil de limpieza (v1, v2, v3),
para ver qué neutraliza cada uno. Además se mide si el atajo "transfiere" a WELFake-dev con la
misma orientación aprendida en ISOT (AUC < 0,5 = el atajo se invierte fuera de ISOT).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from . import plots
from .config import get_paths, load_config
from .corpus import isot_raw, isot_split, welfake_split
from .features import load_lexicon

LOGGER = logging.getLogger(__name__)
_W = re.compile(r"\b\w+\b")


def _per100(pattern: str, flags: int = 0) -> Callable[[str, str], float]:
    rx = re.compile(pattern, flags)

    def f(title: str, body: str) -> float:
        n = max(1, len(_W.findall(body)))
        return 100.0 * len(rx.findall(body)) / n
    return f


def _title_allcaps(title: str, body: str) -> float:
    w = re.findall(r"[A-Za-z]{2,}", title)
    return sum(x.isupper() for x in w) / max(1, len(w))


def _upper_ratio(title: str, body: str) -> float:
    letters = [c for c in body if c.isalpha()]
    return sum(c.isupper() for c in letters) / max(1, len(letters))


def _non_ascii(title: str, body: str) -> float:
    letters = [c for c in body if not c.isspace()]
    return sum(ord(c) > 127 for c in letters) / max(1, len(letters))


def _sent_len(title: str, body: str) -> float:
    sents = [s for s in re.split(r"[.!?]+", body) if s.strip()]
    return len(_W.findall(body)) / max(1, len(sents))


def _attrib(title: str, body: str) -> float:
    lex = load_lexicon("attribution_verbs.txt")
    w = [x.lower() for x in _W.findall(body)]
    return sum(x in lex for x in w) / max(1, len(w))


TEXT_FEATURES: dict[str, Callable[[str, str], float]] = {
    "comillas_curvas_x100p": _per100(r"[’‘“”]"),
    "comillas_rectas_x100p": _per100(r"[\"']"),
    "apostrofo_como_espacio_x100p": _per100(
        r"\b(?:don|doesn|isn|wasn|can|won|didn|aren|weren|it|that|he|she|there|what)\s[ts]\b"),
    "posesivo_con_espacio_x100p": _per100(r"\b[A-Z][a-z]+ s\b"),
    "dias_semana_x100p": _per100(r"\b(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\b"),
    "meses_x100p": _per100(r"\b(?:January|February|March|April|June|July|August|September|October|"
                           r"November|December|Jan\.|Feb\.|Aug\.|Sept\.|Oct\.|Nov\.|Dec\.)"),
    "said_says_told_x100p": _per100(r"\b(?:said|says|told)\b", re.I),
    "verbos_atribucion_ratio": _attrib,
    "dateline_reuters": lambda t, b: float(bool(re.search(r"\(Reuters\)", b))),
    "US_con_puntos_x100p": _per100(r"\bU\.S\."),
    "hora_o_huso_x100p": _per100(r"\b(?:GMT|EDT|EST|[ap]\.m\.)"),
    "exclamaciones_x100p": _per100(r"!"),
    "preguntas_x100p": _per100(r"\?"),
    "parentesis_x100p": _per100(r"\("),
    "guiones_largos_x100p": _per100(r"[—–]"),
    "proporcion_mayusculas": _upper_ratio,
    "titulo_palabras_mayusculas": _title_allcaps,
    "largo_log_palabras": lambda t, b: float(np.log1p(len(_W.findall(b)))),
    "largo_oracion_media": _sent_len,
    "urls_x100p": _per100(r"https?://|\bwww\.", re.I),
    "handles_x100p": _per100(r"@\w+"),
    "hashtags_x100p": _per100(r"#\w+"),
    "xurlx_xuserx_xhashtagx_x100p": _per100(r"\bx(?:url|user|hashtag)x\b"),
    "no_ascii_ratio": _non_ascii,
    "espacio_duro": lambda t, b: float("\xa0" in b),
    "digitos_x100p": _per100(r"\d"),
    "primera_persona_x100p": _per100(r"\b(?:I|we|our|us|my|me)\b"),
    "segunda_persona_x100p": _per100(r"\b(?:you|your)\b", re.I),
}


def _feature_matrix(titles: list[str], bodies: list[str]) -> pd.DataFrame:
    return pd.DataFrame({k: [f(t, b) for t, b in zip(titles, bodies)] for k, f in TEXT_FEATURES.items()})


def _auc(y: np.ndarray, x: np.ndarray) -> float:
    if np.allclose(x, x[0]):
        return 0.5
    return float(roc_auc_score(y, x))


def _stump_acc(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray) -> float:
    clf = DecisionTreeClassifier(max_depth=1, random_state=0).fit(xtr.reshape(-1, 1), ytr)
    return float((clf.predict(xva.reshape(-1, 1)) == yva).mean())


def _split_title(docs: pd.Series) -> tuple[list[str], list[str]]:
    parts = docs.str.split("\n", n=1)
    return parts.str[0].tolist(), [p[1] if len(p) > 1 else p[0] for p in parts]


def run_shortcut_audit() -> pd.DataFrame:
    """Calcula la tabla de atajos y la figura. Usa ISOT train/val y WELFake-dev (nunca test)."""
    paths = get_paths()
    tr, va = isot_split("train", "v1"), isot_split("val", "v1")
    ytr, yva = tr["label"].to_numpy(), va["label"].to_numpy()
    raw_tr, raw_va = isot_raw(tr["id"]), isot_raw(va["id"])
    views = {"crudo": (_feature_matrix(raw_tr["title"].tolist(), raw_tr["text"].tolist()),
                       _feature_matrix(raw_va["title"].tolist(), raw_va["text"].tolist()))}
    for prof in ("v1", "v2", "v3"):
        dtr, dva = isot_split("train", prof)["doc"], isot_split("val", prof)["doc"]
        views[prof] = (_feature_matrix(*_split_title(dtr)), _feature_matrix(*_split_title(dva)))
    wel = welfake_split("dev", "v1")
    wraw = _feature_matrix(wel["title"].tolist(), wel["text"].tolist())
    rows = []
    for feat in TEXT_FEATURES:
        r: dict[str, Any] = {"rasgo": feat}
        xtr_raw = views["crudo"][0][feat].to_numpy()
        sign = 1.0 if _auc(ytr, xtr_raw) >= 0.5 else -1.0
        for name, (ftr, fva) in views.items():
            a = _auc(yva, fva[feat].to_numpy())
            r[f"auc_{name}"] = round(max(a, 1 - a), 4)
            r[f"acc_stump_{name}"] = round(_stump_acc(ftr[feat].to_numpy(), ytr, fva[feat].to_numpy(), yva), 4)
        r["direccion_en_isot"] = "más en falsas" if sign > 0 else "más en reales"
        r["auc_welfake_dev_orientacion_isot"] = round(_auc(wel["label"].to_numpy(), sign * wraw[feat].to_numpy()), 4)
        rows.append(r)
    # Metadatos (no entran al modelo): se reportan para comparar la magnitud del atajo.
    rows.extend(_metadata_rows(tr, va))
    df = pd.DataFrame(rows).sort_values("auc_crudo", ascending=False)
    df.to_csv(paths.reports / "shortcut_audit_v2.csv", index=False)
    tok = token_shortcuts("v2")
    tok.to_csv(paths.reports / "shortcut_tokens_v2.csv", index=False)
    top = df[df["rasgo"].isin(TEXT_FEATURES)].head(16).iloc[::-1]
    plots.hbar_groups(top["rasgo"].tolist(),
                      {"Crudo": top["auc_crudo"].tolist(), "v1": top["auc_v1"].tolist(),
                       "v2": top["auc_v2"].tolist(), "v3": top["auc_v3"].tolist()},
                      "Atajos de un solo rasgo: AUC orientada en validación ISOT", "AUC (0,5 = no predice)",
                      paths.figures / "shortcut_audit_v2.png", xlim=(0.5, 1.0),
                      colors=[plots.NEUTRAL, plots.SERIES[2], plots.SERIES[6], plots.SERIES[3]])
    return df


def _metadata_rows(tr: pd.DataFrame, va: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    rate = tr.groupby("subject")["label"].mean()
    x_tr, x_va = tr["subject"].map(rate).to_numpy(), va["subject"].map(rate).fillna(0.5).to_numpy()
    for name, ftr, fva in (
            ("META subject (no entra al modelo)", x_tr, x_va),
            ("META fecha con espacio final (no entra)", tr["date"].str.endswith(" ").astype(float).to_numpy(),
             va["date"].str.endswith(" ").astype(float).to_numpy()),
            ("META fecha con mes abreviado (no entra)", tr["date"].str.match(r"^[A-Z][a-z]{2}\s").astype(float).to_numpy(),
             va["date"].str.match(r"^[A-Z][a-z]{2}\s").astype(float).to_numpy())):
        a = _auc(va["label"].to_numpy(), fva)
        out.append({"rasgo": name, "auc_crudo": round(max(a, 1 - a), 4),
                    "acc_stump_crudo": round(_stump_acc(ftr, tr["label"].to_numpy(), fva, va["label"].to_numpy()), 4)})
    return out


def token_shortcuts(profile: str, top: int = 40) -> pd.DataFrame:
    """Tokens cuya sola presencia más predice la etiqueta (AUC de un rasgo binario) tras limpiar."""
    tr = isot_split("train", profile)
    y = tr["label"].to_numpy()
    cv = CountVectorizer(binary=True, min_df=50, lowercase=True)
    X = cv.fit_transform(tr["doc"])
    p1 = np.asarray(X[y == 1].mean(axis=0)).ravel()
    p0 = np.asarray(X[y == 0].mean(axis=0)).ravel()
    auc = (p1 - p0 + 1) / 2
    names = cv.get_feature_names_out()
    order = np.argsort(-np.abs(auc - 0.5))[:top]
    return pd.DataFrame({"token": names[order], "auc_presencia": auc[order].round(4),
                         "frac_falsas": p1[order].round(4), "frac_reales": p0[order].round(4)})
