"""Umbral de decisión, zona ambigua y nivel de confianza (labs 06-07; acta, formularios 6 y 7).

Todo se ajusta con el conjunto de VALIDACIÓN. El test nunca participa de estas decisiones.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve

from .metrics import _f1

LABEL_REAL = "probablemente real"
LABEL_FAKE = "probablemente falsa"
LABEL_UNCERTAIN = "incierta, requiere revisión humana"
CONFIDENCE_LEVELS = ("baja", "media", "alta")


def _grid(spec: list[float]) -> np.ndarray:
    start, stop, step = spec
    return np.round(np.arange(start, stop + step / 2, step), 4)


def threshold_table(y: np.ndarray, p: np.ndarray, grid: np.ndarray) -> list[dict[str, float]]:
    """Para cada umbral: precisión, recall y F1 de falsa, y tasa de falsas alarmas (FPR)."""
    y = np.asarray(y).astype(int)
    rows = []
    for t in grid:
        yp = (p >= t).astype(int)
        tp = int(np.sum((yp == 1) & (y == 1)))
        fp = int(np.sum((yp == 1) & (y == 0)))
        fn = int(np.sum((yp == 0) & (y == 1)))
        tn = int(np.sum((yp == 0) & (y == 0)))
        rows.append({"umbral": float(t), "precision_fake": tp / max(1, tp + fp),
                     "recall_fake": tp / max(1, tp + fn), "f1_fake": _f1(y, yp, 1),
                     "fpr": fp / max(1, fp + tn), "fp": fp, "fn": fn})
    return rows


def choose_threshold(y: np.ndarray, p: np.ndarray, grid_spec: list[float], max_fpr: float,
                     tie_tol: float = 0.001) -> dict[str, Any]:
    """Elige el umbral en validación.

    Criterio (acta, formulario 6): maximizar F1 de "falsa" (que equilibra recall y precisión)
    con la restricción de que la tasa de falsas alarmas no supere `max_fpr`. Entre umbrales
    empatados (F1 a menos de `tie_tol` del máximo) se elige el más cercano a 0,5, para no
    mover el corte por diferencias que son ruido.
    """
    table = threshold_table(y, p, _grid(grid_spec))
    feasible = [r for r in table if r["fpr"] <= max_fpr] or table
    best = max(r["f1_fake"] for r in feasible)
    tied = [r for r in feasible if r["f1_fake"] >= best - tie_tol]
    chosen = min(tied, key=lambda r: (abs(r["umbral"] - 0.5), r["umbral"]))
    at_05 = min(table, key=lambda r: abs(r["umbral"] - 0.5))
    return {"umbral": chosen["umbral"], "elegido": chosen, "en_0_5": at_05, "tabla": table,
            "criterio": f"max F1 falsa con FPR <= {max_fpr}; empates (±{tie_tol}) -> más cercano a 0,5"}


def pr_curve(y: np.ndarray, p: np.ndarray) -> dict[str, list[float]]:
    """Curva precisión-recall (clase falsa)."""
    prec, rec, thr = precision_recall_curve(np.asarray(y).astype(int), p)
    return {"precision": prec.tolist(), "recall": rec.tolist(), "umbrales": thr.tolist()}


def is_ambiguous(p: float, low: float, high: float) -> bool:
    """True si la probabilidad cae en la zona ambigua [low, high]."""
    return low <= p <= high


def decide(p: float, threshold: float, low: float, high: float) -> str:
    """Etiqueta final: incierta si cae en la zona ambigua; si no, compara con el umbral."""
    if is_ambiguous(p, low, high):
        return LABEL_UNCERTAIN
    return LABEL_FAKE if p >= threshold else LABEL_REAL


def confidence_score(p: float) -> float:
    """Margen respecto de la decisión: max(p, 1 - p), en [0,5; 1]."""
    return float(max(p, 1.0 - p))


def choose_confidence_cuts(y: np.ndarray, p: np.ndarray, threshold: float, low: float, high: float,
                           grid_spec: list[float], target_acc: float, min_cases: int = 30,
                           window: float = 0.10) -> dict[str, Any]:
    """Cortes de confianza justificados con validación.

    * "baja": la probabilidad cae en la zona ambigua (confianza < `high`).
    * "alta": confianza >= c_alta. Para cada corte candidato c se mide la accuracy LOCAL de los
      casos de validación con confianza en [c, c + `window`) (la última ventana incluye 1,0).
      c_alta es el menor c tal que esa accuracy local es >= `target_acc` en c y en todos los
      cortes mayores (ventanas con menos de `min_cases` casos no se usan para decidir).
    * "media": el resto (entre la zona ambigua y c_alta).

    Se usa la accuracy local (no la acumulada) porque la acumulada queda dominada por la gran
    masa de casos muy seguros y escondería que la franja intermedia acierta menos.
    """
    y = np.asarray(y).astype(int)
    conf = np.maximum(p, 1 - p)
    pred = (p >= threshold).astype(int)
    rows = []
    for c in _grid(grid_spec):
        top = c + window
        m = (conf >= c) & ((conf < top) | (top >= 1.0))
        acc = float((pred[m] == y[m]).mean()) if m.sum() else float("nan")
        rows.append({"corte": float(c), "ventana": [float(c), float(min(1.0, top))], "n": int(m.sum()),
                     "accuracy_local": acc, "decide": bool(m.sum() >= min_cases)})
    c_high = float(_grid(grid_spec)[-1])
    for i, r in enumerate(rows):
        if r["corte"] < high:
            continue
        later = [x for x in rows[i:] if x["decide"]]
        if later and all(x["accuracy_local"] >= target_acc for x in later):
            c_high = r["corte"]
            break
    cuts = {"baja_si_conf_menor_a": float(high), "alta_si_conf_mayor_o_igual_a": c_high}
    return {"cortes": cuts, "tabla": rows, "objetivo_accuracy_alta": target_acc,
            "criterio": f"accuracy local (ventana {window}) >= {target_acc} desde el corte en adelante",
            "por_nivel_val": accuracy_by_level(y, p, threshold, cuts)}


def confidence_level(p: float, cuts: dict[str, float]) -> str:
    """Nivel de confianza (baja/media/alta) a partir de la probabilidad calibrada."""
    c = confidence_score(p)
    if c < cuts["baja_si_conf_menor_a"]:
        return "baja"
    if c >= cuts["alta_si_conf_mayor_o_igual_a"]:
        return "alta"
    return "media"


def accuracy_by_level(y: np.ndarray, p: np.ndarray, threshold: float, cuts: dict[str, float]
                      ) -> dict[str, dict[str, float]]:
    """Accuracy y cantidad de casos por nivel de confianza."""
    y = np.asarray(y).astype(int)
    levels = np.array([confidence_level(float(x), cuts) for x in p])
    pred = (p >= threshold).astype(int)
    out = {}
    for lev in CONFIDENCE_LEVELS:
        m = levels == lev
        out[lev] = {"n": int(m.sum()), "fraccion": float(m.mean()),
                    "accuracy": float((pred[m] == y[m]).mean()) if m.sum() else float("nan")}
    return out


def coverage_curve(y: np.ndarray, p: np.ndarray, threshold: float,
                   half_widths: np.ndarray | None = None) -> list[dict[str, float]]:
    """Cobertura vs. accuracy: se deriva a revisión todo lo que cae en [0,5-w; 0,5+w]."""
    y = np.asarray(y).astype(int)
    half_widths = np.round(np.arange(0, 0.46, 0.05), 2) if half_widths is None else half_widths
    pred = (p >= threshold).astype(int)
    rows = []
    for w in half_widths:
        keep = ~((p >= 0.5 - w) & (p <= 0.5 + w)) if w > 0 else np.ones_like(y, dtype=bool)
        rows.append({"semiancho": float(w), "zona": [float(0.5 - w), float(0.5 + w)],
                     "cobertura": float(keep.mean()),
                     "accuracy_cubiertos": float((pred[keep] == y[keep]).mean()) if keep.any() else float("nan"),
                     "derivados_a_revision": int((~keep).sum())})
    return rows


def zone_summary(y: np.ndarray, p: np.ndarray, threshold: float, low: float, high: float
                 ) -> dict[str, float]:
    """Efecto de la zona ambigua configurada: fracción derivada y accuracy del resto."""
    y = np.asarray(y).astype(int)
    amb = (p >= low) & (p <= high)
    pred = (p >= threshold).astype(int)
    return {"zona": [low, high], "fraccion_derivada": float(amb.mean()), "n_derivados": int(amb.sum()),
            "accuracy_sin_zona": float((pred == y).mean()),
            "accuracy_resto": float((pred[~amb] == y[~amb]).mean()) if (~amb).any() else float("nan"),
            "accuracy_en_zona": float((pred[amb] == y[amb]).mean()) if amb.any() else float("nan")}
