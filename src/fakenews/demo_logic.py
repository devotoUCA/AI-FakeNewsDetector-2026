"""Lógica de la demo, separada de Streamlit para poder testearla sin levantar el servidor."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import get_paths
from .predict import DISCLAIMER, ModelNotTrainedError, Predictor

PERMANENT_NOTICE = (
    "⚠️ " + DISCLAIMER + " Solo inglés. Puede fallar con sátira, noticias nuevas o temas fuera del "
    "dataset. Nunca debe usarse como único criterio para moderar contenido."
)
LEVEL_TEXT = {"alta": "Alta", "media": "Media", "baja": "Baja"}


def load_predictor() -> tuple[Predictor | None, str | None]:
    """Intenta cargar el modelo; si falta, devuelve un mensaje con el comando para entrenarlo."""
    try:
        return Predictor(), None
    except ModelNotTrainedError as exc:
        return None, f"{exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"No se pudo cargar el modelo ({type(exc).__name__}: {exc}). Probá `make train`."


def analyze(predictor: Predictor, text: str) -> dict[str, Any]:
    """Arma todo lo que la demo muestra para un texto (sin depender de Streamlit)."""
    out = predictor.predict(text, with_similar=True)
    p = out["p_false"]
    return {
        "titulo_resultado": out["label"].capitalize(),
        "p_false": p,
        "p_false_texto": "no disponible" if p is None else f"{p:.1%}",
        "barra": 0.0 if p is None else float(p),
        "confianza": LEVEL_TEXT[out["confidence_level"]],
        "zona_ambigua": bool(out["ambiguous"]),
        "aviso_zona": ("El resultado cae en la zona ambigua o la entrada está fuera de distribución: "
                       "requiere revisión humana.") if out["ambiguous"] else "",
        "terminos_falsa": out["top_terms"]["hacia_falsa"],
        "terminos_real": out["top_terms"]["hacia_real"],
        "advertencias": out["warnings"],
        "similares": out.get("similar_examples", []),
        "version": out["model_version"],
        "crudo": out,
    }


def example_texts(n_per_class: int = 2) -> list[dict[str, str]]:
    """Ejemplos precargados: textos del propio conjunto de TEST (marcados como tales)."""
    path = get_paths().processed / "test.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, keep_default_na=False)
    df = df[(df["n_words"] > 120) & (df["n_words"] < 600)]
    rows = []
    for lab, name in ((0, "real"), (1, "falsa")):
        sub = df[df["label"] == lab].sample(n_per_class, random_state=7)
        for _, r in sub.iterrows():
            rows.append({"nombre": f"[Ejemplo del dataset de test · etiqueta {name}] {r['doc'].split(chr(10))[0][:70]}",
                         "texto": r["doc"]})
    return rows
