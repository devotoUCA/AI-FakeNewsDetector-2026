"""Descubrimiento automático de atajos de FORMATO (iteración 3, punto A2).

El texto se mapea a clases de carácter (minúscula -> a, mayúscula -> A, dígito -> 0, espacio -> _,
salto de línea -> ¶, espacio duro -> ⍽; la puntuación y los símbolos Unicode quedan tal cual) y se
miden TODOS los n-gramas de 1 a 3 clases: así salen solos patrones como "a.A" (oración pegada),
"__" (doble espacio), "_." (espacio antes del punto) o "—" (guion largo), sin listarlos a mano.

Para cada patrón se calcula su AUC para predecir la etiqueta en ISOT-train y su AUC para
distinguir ISOT de los externos de desarrollo. Nunca usa el test ni los holdouts.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import CountVectorizer

from . import plots
from .config import get_paths, load_config
from .corpus import isot_raw, isot_split, liar_split, welfake_split

LOGGER = logging.getLogger(__name__)
MIN_DF = 0.005
_TRANS = {}
for _c in "abcdefghijklmnopqrstuvwxyz":
    _TRANS[ord(_c)] = "a"
    _TRANS[ord(_c.upper())] = "A"
for _c in "0123456789":
    _TRANS[ord(_c)] = "0"
_TRANS[ord(" ")] = "_"
_TRANS[ord("\n")] = "¶"
_TRANS[ord("\t")] = "⇥"
_TRANS[0xA0] = "⍽"
DESCRIPTIONS = {
    "a.A": "oración pegada: punto sin espacio y mayúscula", "a.a": "punto sin espacio y minúscula",
    "a,a": "coma sin espacio", "a?A": "signo ? sin espacio", "a!A": "signo ! sin espacio", "__": "doble espacio",
    "_.": "espacio antes del punto", "_,": "espacio antes de la coma", "_._": "punto aislado", "_,_": "coma aislada",
    "…": "puntos suspensivos Unicode", "...": "puntos suspensivos ASCII", "—": "guion largo", "–": "guion medio",
    "_-_": "guion corto aislado", "⍽": "espacio duro", "¶": "salto de línea", ".A": "mayúscula tras punto sin espacio",
    "._A": "mayúscula tras punto con espacio", "(_": "espacio tras paréntesis", "_)": "espacio antes de paréntesis",
}


def to_format_classes(text: str) -> str:
    """Mapea un texto a clases de carácter."""
    return (text or "").translate(_TRANS)


def _is_format(pattern: str) -> bool:
    return bool(pattern.strip("aA0"))


def auc_columns(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUC de cada columna (vectorizado por rangos, con empates promediados)."""
    ranks = rankdata(X, axis=0)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return (ranks[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def _rates(vec: CountVectorizer, texts: list[str]) -> np.ndarray:
    X = vec.transform(texts).toarray().astype(np.float32)
    lens = np.maximum(1, np.array([len(t) for t in texts], dtype=np.float32))[:, None]
    return 100.0 * X / lens


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df.sample(min(n, len(df)), random_state=seed)


def discover(view: str, tr: pd.DataFrame, in_val: list[str], ext: list[str]) -> pd.DataFrame:
    """Tabla de patrones de una vista (texto crudo o limpio)."""
    train_txt = [to_format_classes(t) for t in tr["txt"]]
    vec = CountVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False, min_df=MIN_DF)
    vec.fit(train_txt)
    names = vec.get_feature_names_out()
    keep = np.array([_is_format(n) for n in names])
    y = tr["label"].to_numpy()
    R = _rates(vec, train_txt)[:, keep]
    a_label = auc_columns(R, y)
    Rin = _rates(vec, [to_format_classes(t) for t in in_val])[:, keep]
    Rex = _rates(vec, [to_format_classes(t) for t in ext])[:, keep]
    a_dom = auc_columns(np.vstack([Rin, Rex]), np.r_[np.ones(len(Rin), int), np.zeros(len(Rex), int)])
    names = names[keep]
    return pd.DataFrame({
        "vista": view, "patron": names, "descripcion": [DESCRIPTIONS.get(n, "") for n in names],
        "auc_etiqueta": np.round(np.maximum(a_label, 1 - a_label), 4),
        "mas_frecuente_en": np.where(a_label >= 0.5, "falsas", "reales"),
        "por_100c_falsas": np.round(R[y == 1].mean(axis=0), 4), "por_100c_reales": np.round(R[y == 0].mean(axis=0), 4),
        "auc_dominio_isot_vs_externos": np.round(np.maximum(a_dom, 1 - a_dom), 4)})


def run_format_discovery(profiles: tuple[str, ...] = ("v2", "v2_1")) -> pd.DataFrame:
    """Corre el descubrimiento (crudo y por perfil) y guarda CSV y figura."""
    cfg, paths = load_config(), get_paths()
    seed = cfg["seed"]
    n = cfg.get("format_discovery", {}).get("sample", 8000)
    frames = []
    tr0 = _sample(isot_split("train", "v1"), n, seed)
    va0 = _sample(isot_split("val", "v1"), 2000, seed)
    wel = _sample(welfake_split("dev", "v1"), 2000, seed)
    lv = liar_split("valid", "v1")
    raw_tr, raw_va = isot_raw(tr0["id"]), isot_raw(va0["id"])
    tr_raw = tr0.assign(txt=(raw_tr["title"] + "\n" + raw_tr["text"]).to_numpy())
    ext_raw = (wel["title"] + "\n" + wel["text"]).tolist() + lv["statement"].tolist()
    frames.append(discover("crudo", tr_raw, (raw_va["title"] + "\n" + raw_va["text"]).tolist(), ext_raw))
    for prof in profiles:
        tr = isot_split("train", prof).set_index("id").loc[tr0["id"]].reset_index()
        va = isot_split("val", prof).set_index("id").loc[va0["id"]]
        w = welfake_split("dev", prof).set_index("id").loc[wel["id"]]
        ext = w["doc"].tolist() + liar_split("valid", prof)["doc"].tolist()
        frames.append(discover(prof, tr.assign(txt=tr["doc"]), va["doc"].tolist(), ext))
    df = pd.concat(frames, ignore_index=True).sort_values(["vista", "auc_etiqueta"], ascending=[True, False])
    df.to_csv(paths.reports / "format_shortcut_discovery_v3.csv", index=False)
    last = profiles[0]
    top = df[df["vista"] == last].head(14)["patron"].tolist()[::-1]

    def vals(view: str) -> list[float]:
        d = df[df["vista"] == view].set_index("patron")["auc_etiqueta"]
        return [float(d.get(p, 0.5)) for p in top]

    series = {v: vals(v) for v in ["crudo", *profiles]}
    plots.hbar_groups([repr(p)[1:-1] for p in top], series,
                      f"Atajos de formato (top según {last}): AUC para predecir la etiqueta", "AUC (0,5 = no predice)",
                      paths.figures / "format_shortcuts_v3.png", xlim=(0.5, 1.0),
                      colors=[plots.NEUTRAL, plots.SERIES[6], plots.SERIES[3]][:len(series)])
    return df
