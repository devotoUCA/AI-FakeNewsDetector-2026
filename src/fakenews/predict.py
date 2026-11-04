"""API única de inferencia: la usan la CLI, la demo y los tests.

`predict(text)` devuelve un diccionario con: label, p_false, confidence_level, confidence,
ambiguous, top_terms, warnings (y extras: similar_examples, model_version, style).
Nunca lanza excepciones por la entrada: ante cualquier problema devuelve "incierta" con
una advertencia explicativa (lab 21: contrato de entrada/salida y validación de entradas).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .cleaning import clean_text, word_count
from .config import get_paths, load_config
from .explain import style_part, style_summary, term_contributions
from .features import english_stopword_ratio, known_vocab_ratio, non_ascii_letter_ratio
from .models import p_fake
from .thresholds import (LABEL_UNCERTAIN, confidence_level, confidence_score, decide,
                         is_ambiguous)

LOGGER = logging.getLogger(__name__)
DISCLAIMER = ("Estimación basada en patrones del texto, no un veredicto ni una verificación de "
              "hechos. La probabilidad refleja parecido con los textos de entrenamiento (noticias "
              "políticas de EE.UU., 2015-2018), no certeza sobre la verdad de los hechos.")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)


class ModelNotTrainedError(FileNotFoundError):
    """No hay modelo entrenado en `models/`."""


class Predictor:
    """Carga los artefactos una vez y responde predicciones."""

    def __init__(self, models_dir: Path | None = None, load_retrieval: bool = True) -> None:
        models_dir = models_dir or get_paths().models
        path = models_dir / "final_model.joblib"
        if not path.exists():
            raise ModelNotTrainedError(
                f"No se encontró {path}. Entrená el modelo con `make train` "
                "(o `python -m fakenews.cli train`).")
        self.artifact: dict[str, Any] = joblib.load(path)
        self.model = self.artifact["model"]
        self.threshold: float = float(self.artifact["threshold"])
        self.low, self.high = map(float, self.artifact["ambiguous"])
        self.cuts: dict[str, float] = self.artifact["confidence_cuts"]
        self.vocabulary = self.artifact["vocabulary"]
        self.version: str = self.artifact["metadata"]["version"]
        self.retrieval = None
        rpath = models_dir / "retrieval.joblib"
        if load_retrieval and rpath.exists():
            self.retrieval = joblib.load(rpath)
        self.ood = load_config()["ood"]

    # ----------------------------------------------------------------------------------
    def input_warnings(self, raw: str, cleaned: str) -> tuple[list[str], bool, bool]:
        """Advertencias de entrada fuera de distribución.

        Returns:
            (advertencias, forzar_incierta, bajar_confianza)
        """
        o = self.ood
        warns: list[str] = []
        force, lower = False, False
        n = word_count(cleaned)
        if not raw.strip():
            return ["El texto está vacío."], True, True
        if n == 0:
            if _URL.search(raw):
                return ["El texto solo contiene enlaces: el modelo analiza texto, no sigue links."], True, True
            return ["No quedó texto analizable después de limpiar (solo símbolos o marcas)."], True, True
        if n < o["min_words"]:
            warns.append(f"Texto muy corto ({n} palabras; mínimo recomendado {o['min_words']}). "
                         "El modelo se entrenó con artículos completos.")
            force = True
        if n > o["max_words"]:
            warns.append(f"Texto muy largo ({n} palabras): está fuera del rango típico del "
                         "entrenamiento; la confianza se reduce.")
            lower = True
        if (english_stopword_ratio(cleaned) < o["min_english_stopword_ratio"]
                or non_ascii_letter_ratio(cleaned) > o["max_non_ascii_letter_ratio"]):
            warns.append("El texto no parece estar en inglés. El modelo solo fue entrenado con "
                         "noticias en inglés: el resultado no es interpretable.")
            force = True
        kv = known_vocab_ratio(cleaned, self.vocabulary)
        if kv < o["min_known_vocab_ratio"]:
            warns.append(f"Solo el {kv:.0%} de las palabras es conocido por el modelo: el texto "
                         "está lejos de los datos de entrenamiento.")
            force = True
        return warns, force, lower

    def similar_examples(self, cleaned: str, k: int) -> list[dict[str, Any]]:
        """Noticias de entrenamiento más parecidas (coseno), como trazabilidad (lab 17)."""
        if self.retrieval is None or not cleaned:
            return []
        pipe = self.retrieval["pipeline"]
        x = pipe.named_steps["tfidf"].transform([cleaned])
        if x.nnz == 0:
            return []
        dist, idx = pipe.named_steps["clf"].kneighbors(x, n_neighbors=k)
        return [{"titulo": self.retrieval["titles"][i],
                 "etiqueta": "falsa" if self.retrieval["labels"][i] == 1 else "real",
                 "similitud": round(float(1 - d), 3)} for d, i in zip(dist[0], idx[0])]

    def predict(self, text: object, with_similar: bool = False) -> dict[str, Any]:
        """Diagnóstico de un texto. Nunca lanza excepción por la entrada."""
        raw = text if isinstance(text, str) else ""
        base: dict[str, Any] = {"label": LABEL_UNCERTAIN, "p_false": None, "confidence_level": "baja",
                                "confidence": None, "ambiguous": True,
                                "top_terms": {"hacia_falsa": [], "hacia_real": []}, "warnings": [],
                                "similar_examples": [], "model_version": self.version,
                                "disclaimer": DISCLAIMER}
        if not isinstance(text, str):
            base["warnings"].append("La entrada no es texto.")
            return base
        try:
            cleaned = clean_text(raw)
            warns, force, lower = self.input_warnings(raw, cleaned)
            base["warnings"] = warns
            if not cleaned or word_count(cleaned) == 0:
                return base
            p = float(p_fake(self.model, [cleaned])[0])
            amb = is_ambiguous(p, self.low, self.high)
            label = LABEL_UNCERTAIN if force else decide(p, self.threshold, self.low, self.high)
            level = confidence_level(p, self.cuts)
            if force:
                level = "baja"
            elif lower and level != "baja":
                level = "media" if level == "alta" else "baja"
            out = {**base, "label": label, "p_false": round(p, 4), "confidence_level": level,
                   "confidence": round(confidence_score(p), 4), "ambiguous": bool(amb or force),
                   "top_terms": term_contributions(self.model, cleaned, top_k=8)}
            if style_part(self.model):
                out["style"] = style_summary(cleaned)
            if with_similar:
                out["similar_examples"] = self.similar_examples(
                    cleaned, int(load_config()["demo"]["similar_examples"]))
            self._log(raw, out)
            return out
        except Exception as exc:  # noqa: BLE001 - la inferencia nunca debe romper la demo
            LOGGER.exception("Error en predict")
            base["warnings"].append(f"Error interno al analizar el texto ({type(exc).__name__}). "
                                    "El resultado no está disponible.")
            return base

    def _log(self, raw: str, out: dict[str, Any]) -> None:
        """Registro opcional (sin guardar el texto: solo su hash) para monitoreo (lab 21)."""
        if not load_config()["demo"].get("log_predictions"):
            return
        rec = {"ts": dt.datetime.now().isoformat(timespec="seconds"),
               "sha256_texto": hashlib.sha256(raw.encode()).hexdigest(), "model": self.version,
               "label": out["label"], "p_false": out["p_false"], "warnings": len(out["warnings"])}
        with (get_paths().reports / "predictions_log.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Predictor por defecto (cacheado)."""
    return Predictor()


def predict(text: object, with_similar: bool = False) -> dict[str, Any]:
    """Atajo: `predict(text) -> {label, p_false, confidence_level, ambiguous, top_terms, warnings}`."""
    return get_predictor().predict(text, with_similar=with_similar)


def predict_batch_p(texts: list[str]) -> np.ndarray:
    """Probabilidades crudas del modelo productivo para una lista de textos ya limpios."""
    return p_fake(get_predictor().model, texts)
