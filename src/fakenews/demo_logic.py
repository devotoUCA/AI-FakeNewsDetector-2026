"""Lógica de la demo, separada de Streamlit para poder testearla sin levantar el servidor."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import get_paths
from .predict import DISCLAIMER, ModelNotTrainedError, Predictor, display_probability

PERMANENT_NOTICE = (
    "⚠️ " + DISCLAIMER + " Solo inglés. Puede fallar con sátira, noticias nuevas o temas fuera del "
    "dataset. Nunca debe usarse como único criterio para moderar contenido."
)
LEVEL_TEXT = {"alta": "Alta", "media": "Media", "baja": "Baja"}
OUT_OF_DOMAIN_MESSAGE = ("Este texto no se parece a las noticias con las que se entrenó el modelo: "
                         "no se emite veredicto.")


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
    inside = bool(out.get("in_domain", p is not None))
    ref = out.get("p_false_reference")
    return {
        "titulo_resultado": out["label"].capitalize() if inside else "Sin veredicto: fuera del dominio de validez",
        "en_dominio": inside,
        "mensaje_dominio": "" if inside else OUT_OF_DOMAIN_MESSAGE,
        "p_false": p,
        "p_false_texto": out.get("p_false_display", "no disponible"),
        "barra": 0.0 if p is None else float(p),
        "referencia_texto": None if ref is None else display_probability(float(ref)),
        "confianza": LEVEL_TEXT[out["confidence_level"]],
        "zona_ambigua": bool(out["ambiguous"]) and inside,
        "aviso_zona": "El resultado cae en la zona ambigua: requiere revisión humana." if out["ambiguous"] and inside else "",
        "terminos_falsa": out["top_terms"]["hacia_falsa"] if inside else [],
        "terminos_real": out["top_terms"]["hacia_real"] if inside else [],
        "advertencias": out["warnings"],
        "similares": out.get("similar_examples", []),
        "version": out["model_version"],
        "perfil": out.get("cleaning_profile", "v1"),
        "puntaje_dominio": out.get("domain_score"),
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
    extra = get_paths().root / "examples" / "03_afirmacion_liar_valid.txt"
    if extra.exists():
        rows.append({"nombre": "[Ejemplo fuera de dominio · afirmación corta de LIAR-valid]",
                     "texto": extra.read_text(encoding="utf-8")})
    return rows
