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
from .domain import domain_score
from .models import p_fake
from .ood import ood_features, ood_score, shrink
from .thresholds import (LABEL_UNCERTAIN, confidence_level, confidence_score, decide,
                         is_ambiguous)

LOGGER = logging.getLogger(__name__)
DISCLAIMER = ("Estimación basada en patrones del texto, no un veredicto ni una verificación de "
              "hechos. La probabilidad refleja parecido con los textos de entrenamiento (noticias "
              "políticas de EE.UU., 2015-2018), no certeza sobre la verdad de los hechos.")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_NEUTRAL = re.compile(r"\bx(?:url|user|hashtag|num|time)x\b")


OUT_OF_DOMAIN = "fuera_de_dominio"
P_MIN, P_MAX = 0.001, 0.999


def clip_public(p: float) -> float:
    """Probabilidad para la salida pública: nunca 0 ni 1 literales, nunca NaN."""
    if p != p:  # NaN
        return 0.5
    return round(min(P_MAX, max(P_MIN, float(p))), 4)


def display_probability(p: float) -> str:
    """Texto de la probabilidad: extremos como "< 1 %" y "> 99 %"."""
    if p < 0.01:
        return "< 1 %"
    if p > 0.99:
        return "> 99 %"
    return f"{100 * p:.1f} %".replace(".", ",")


class ModelNotTrainedError(FileNotFoundError):
    """No hay modelo entrenado en `models/`."""


class Predictor:
    """Carga los artefactos una vez y responde predicciones."""

    def __init__(self, models_dir: Path | None = None, load_retrieval: bool = True,
                 artifact: str | None = None) -> None:
        models_dir = models_dir or get_paths().models
        inf = load_config().get("inference", {})
        wanted = artifact or inf.get("artifact", "final_model.joblib")
        path = models_dir / wanted
        if not path.exists() and artifact is None:
            path = models_dir / "final_model.joblib"  # compatibilidad con la iteración 1
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
        # Perfil de limpieza con el que se entrenó (iteración 1 = v1): se usa igual al predecir.
        self.profile: str = self.artifact.get("profile", "v1")
        self.ood_params: dict[str, Any] | None = self.artifact.get("ood")
        # Compuerta de dominio de validez (iteración 3); los artefactos anteriores no la tienen.
        self.gate: dict[str, Any] | None = self.artifact.get("domain_gate")
        self.retrieval = None
        rname = {2: "retrieval_v2.joblib", 3: "retrieval_v3.joblib"}.get(self.artifact.get("iteration"),
                                                                        "retrieval.joblib")
        rpath = models_dir / rname
        if (load_retrieval or self.ood_params or self.gate) and rpath.exists():
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
        # Los tokens neutros de v2 (xurlx, xnumx, ...) no cuentan como palabras de contenido.
        content = _NEUTRAL.sub(" ", cleaned)
        n = word_count(content)
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
        if n >= 8 and (english_stopword_ratio(cleaned) < o["min_english_stopword_ratio"]
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
        base.update({"p_false_raw": None, "ood_score": None, "cleaning_profile": self.profile,
                     "in_domain": False, "domain_score": None, "p_false_reference": None,
                     "p_false_display": "no disponible"})
        if not isinstance(text, str):
            base["warnings"].append("La entrada no es texto.")
            return base
        try:
            cleaned = clean_text(raw, profile=self.profile)
            warns, force, lower = self.input_warnings(raw, cleaned)
            base["warnings"] = warns
            if not cleaned or word_count(_NEUTRAL.sub(" ", cleaned)) == 0:
                base["warnings"].append(f"{OUT_OF_DOMAIN}: no hay texto analizable.")
                return base
            p = float(p_fake(self.model, [cleaned])[0])
            if self.gate is not None:
                return self._predict_with_gate(raw, cleaned, p, warns, force, lower, base, with_similar)
            p_raw, s = p, None
            cap = False
            if self.ood_params and self.retrieval is not None:
                s = float(ood_score(ood_features([cleaned], self.retrieval, self.vocabulary), self.ood_params)[0])
                p = float(shrink(np.array([p]), np.array([s]), self.ood_params["prior"])[0])
                if s >= self.ood_params["tau"]:
                    warns.append(f"Fuera de distribución (puntaje OOD {s:.2f} >= {self.ood_params['tau']:.2f}): "
                                 "el texto se parece poco a las noticias de entrenamiento. Se responde "
                                 "'incierta' y la probabilidad se contrajo hacia la proporción base.")
                    force = True
                elif s >= self.ood_params["tau_cap"]:
                    warns.append(f"El texto se parece menos que lo habitual al entrenamiento (puntaje OOD "
                                 f"{s:.2f}): la confianza se limita a 'media'.")
                    cap = True
            amb = is_ambiguous(p, self.low, self.high)
            label = LABEL_UNCERTAIN if force else decide(p, self.threshold, self.low, self.high)
            level = confidence_level(p, self.cuts)
            if force:
                level = "baja"
            elif lower and level != "baja":
                level = "media" if level == "alta" else "baja"
            if cap and level == "alta":
                level = "media"
            out = {**base, "label": label, "p_false": round(p, 4), "confidence_level": level,
                   "p_false_raw": round(p_raw, 4), "ood_score": None if s is None else round(s, 4),
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

    def _predict_with_gate(self, raw: str, cleaned: str, p: float, warns: list[str], force: bool, lower: bool,
                           base: dict[str, Any], with_similar: bool) -> dict[str, Any]:
        """Iteración 3: dentro del dominio, probabilidad calibrada; fuera, sin veredicto."""
        score = float(domain_score([cleaned], self.gate, self.retrieval, self.vocabulary)[0])
        inside = score >= self.gate["umbral"] and not force
        pub = clip_public(p)
        out = {**base, "in_domain": bool(inside), "domain_score": round(score, 4), "p_false_raw": pub,
               "top_terms": term_contributions(self.model, cleaned, top_k=8)}
        if not inside:
            why = ("el texto no cumple las condiciones de entrada (ver las otras advertencias)" if force else
                   f"no se parece a las noticias de entrenamiento (puntaje de dominio {score:.2f} < "
                   f"{self.gate['umbral']:.2f})")
            warns.append(f"{OUT_OF_DOMAIN}: {why}. No se emite veredicto.")
            out.update({"label": LABEL_UNCERTAIN, "p_false": None, "confidence_level": "baja", "confidence": None,
                        "ambiguous": True, "p_false_reference": pub, "p_false_display": "sin veredicto"})
        else:
            level = confidence_level(p, self.cuts)
            if lower and level != "baja":
                level = "media" if level == "alta" else "baja"
            out.update({"label": decide(p, self.threshold, self.low, self.high), "p_false": pub,
                        "confidence_level": level, "confidence": round(confidence_score(pub), 4),
                        "ambiguous": bool(is_ambiguous(p, self.low, self.high)), "p_false_reference": pub,
                        "p_false_display": display_probability(p)})
        if with_similar:
            out["similar_examples"] = self.similar_examples(cleaned, int(load_config()["demo"]["similar_examples"]))
        self._log(raw, out)
        return out

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
