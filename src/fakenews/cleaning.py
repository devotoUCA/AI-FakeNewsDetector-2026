"""Limpieza de texto: una única función pura, compartida por entrenamiento e inferencia.

Hay dos modos:

* ``"full"`` (el que usa el modelo): quita marcas de fuente, de formato y de plataforma
  que separan las clases del dataset ISOT sin decir nada sobre la veracidad del contenido
  (dateline de agencia, nombre de la agencia, créditos de fotos, embeds de tuits, URLs,
  textos promocionales repetidos, etiquetas "(VIDEO)" en títulos y la codificación de
  comillas/apóstrofos, que difiere entre ambas fuentes).
* ``"minimal"``: solo normaliza unicode y espacios. Se usa únicamente en el experimento
  controlado de fuga (Fase 5), para medir cuánto dependía el modelo de esas marcas.

Cada regla está justificada con conteos reales en ``reports/leakage_audit.json``.
Las minúsculas las aplica el vectorizador TF-IDF; acá se conserva la caja porque los
rasgos de estilo (proporción de mayúsculas) la necesitan.
"""

from __future__ import annotations

import html
import re
import unicodedata

CLEAN_MODES = ("full", "minimal")

_MONTHS = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
           r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")

# --- Reglas de reemplazo, en orden. Cada una: (nombre, patrón compilado, reemplazo) ---------
_RULES: list[tuple[str, re.Pattern[str], str]] = [
    # HTML residual
    ("html_tag", re.compile(r"<[^<>]{1,200}>"), " "),
    # Dateline de agencia: "WASHINGTON (Reuters) -", "CHENGDU, China (Reuters) -", "(Reuters) -"
    ("agency_dateline", re.compile(
        r"(?:\b[A-Z][A-Z\.\-'’]+(?:[ /][A-Z][A-Z\.\-'’]+)*(?:,\s*[A-Z][A-Za-z\.]{1,20}){0,2}\s*)?"
        r"\((?:Reuters|AP|AFP)\)\s*[-–—]*\s*"), " "),
    # Avisos de corrección de agencia al inicio: "(In Dec. 25 story, ... corrects ...)"
    ("agency_correction", re.compile(
        r"\((?:[^()]{0,250}?)\b(?:corrects?|refiles?|updates? to|adds?)\b[^()]{0,250}\)"), " "),
    # Plantilla de Reuters para tuits de Trump y descargo de SAP (texto repetido idéntico)
    ("reuters_template", re.compile(
        r"The following statements\s+were posted to the verified Twitter accounts of U\.S\. "
        r"President Donald Trump, @realDonaldTrump and @POTUS\.\s*"
        r"(?:The opinions expressed are his own\.\s*)?"
        r"(?:Reuters has not edited the statements or confirmed their accuracy\.\s*)?", re.I), " "),
    ("reuters_template_tail", re.compile(
        r"(?:The opinions expressed are his own\.|Reuters has not edited the statements or "
        r"confirmed their accuracy\.)", re.I), " "),
    ("sponsor_disclaimer", re.compile(
        r"(?:This (?:article|story) was funded in part by SAP\.?|It was independently created by "
        r"(?:the )?Reuters[’']? editorial staff[^.]{0,120}\.|SAP had no editorial involvement in "
        r"its creation or production\.)", re.I), " "),
    ("daily_highlights", re.compile(r"Highlights of the day for U\.S\. President Donald Trump[’']?s "
                                    r"administration on [A-Z][a-z]+:?", re.I), " "),
    # Plantilla de Reuters para tuits: "[0804 EST] Source link: (bit.ly/...)"
    ("reuters_tweet_stamp", re.compile(r"\[\s*\d{3,4}\s*E[SD]T\s*\]|\bSource link\s*:?(?:\s*\(\s*[^()]{0,80}\))*",
                                       re.I), " "),
    # Nombre de la agencia en cualquier lugar ("told Reuters", "Reuters/Ipsos")
    ("agency_name", re.compile(r"\b(?:Thomson\s+)?Reuters\b(?:[’']s)?", re.I), " "),
    # Embeds de tuits: "— Name (@handle) December 31, 2017" y pic.twitter.com/...
    ("pic_twitter", re.compile(r"\bpic\.twitter\.com/\S+", re.I), " "),
    ("tweet_signature", re.compile(r"\(@\w{1,30}\)\s*" + _MONTHS + r"\s+\d{1,2},?\s+\d{4}"), " "),
    # URLs y dominios
    ("url", re.compile(r"(?:https?://|\bwww\.)\S*", re.I), " "),
    ("domain", re.compile(r"\b[\w\-]+(?:\.[\w\-]+)*\.(?:com|org|net|co|ly|gov|edu|tv|info|us)"
                          r"(?:/\S*)?\b", re.I), " "),
    ("handle", re.compile(r"@\w{0,30}"), " "),
    # Créditos de imagen: "Featured image via Flickr", "Photo by X/Getty Images"
    ("featured_image", re.compile(r"Featured\s+image\w*[^.\n]{0,150}\.?", re.I), " "),
    ("photo_credit", re.compile(
        r"\b(?:Photo|Image|Screenshot|Screen capture)s?\s+(?:by|via|credit:?|courtesy of)\s+"
        r"[^.\n]{0,100}?(?:Getty Images|/Getty|AP Photo|AFP|Flickr|Twitter|YouTube|Wikimedia"
        r"(?: Commons)?|screen ?(?:shot|capture|grab))\.?", re.I), " "),
    ("getty", re.compile(r"\b[\w\-]*/?\s?Getty\s+Images\b\.?", re.I), " "),
    # Atribuciones de blog: "Via: Breitbart News", "Read more: Daily Mail"
    ("via_line", re.compile(r"\b(?:Via|h/t|Read more|READ MORE|Source)\s*:\s*[^.\n]{0,60}", re.I), " "),
    ("read_more", re.compile(r"\bREAD MORE\b[^.\n]{0,80}"), " "),
    # Etiquetas de medios en títulos: "(VIDEO)", "[TWEETS]", "WATCH:"
    ("media_tag", re.compile(
        r"[\(\[]\s*(?:VIDEOS?|IMAGES?|PHOTOS?|PICS?|TWEETS?|AUDIO|GRAPHIC|SCREENSHOTS?|DETAILS|"
        r"LIVE|LISTEN|WATCH|GIF)\b[^\)\]]{0,40}[\)\]]", re.I), " "),
    ("media_prefix", re.compile(r"(?m)^\s*(?:WATCH|LISTEN|VIDEO|WATCH LIVE)\s*:\s*"), " "),
    # Instrucciones de video embebido (plataforma de blog): "Watch the video below:", "Watch:"
    ("video_embed", re.compile(
        r"\b(?:watch|see|here s|here is|check out|take a look at)\b[^.!?\n]{0,50}?\b(?:video|clip|"
        r"footage|segment|interview)s?\b[^.!?\n]{0,30}?\b(?:below|here|above)\b\s*:?"
        r"|\b(?:watch|listen)\s*(?:here|below|it here)?\s*:", re.I), " "),
    # Prefijos de formato de agencia en títulos: "Factbox:", "Exclusive:"
    ("agency_title_prefix", re.compile(
        r"(?m)^\s*(?:Factbox|Exclusive|Explainer|Timeline|Highlights|WRAPUP\s*\d*|"
        r"Breakingviews|Instant View)\s*:\s*", re.I), " "),
    # Comillas y apóstrofos: las reales usan ’ “ ” y en las falsas el apóstrofo suele estar
    # reemplazado por un espacio ("don t"). Se unifica todo a espacio.
    ("quotes", re.compile(r"[\"“”„«»‘’‚‛`´']"), " "),
    # Caracteres de control y de ancho cero
    ("control_chars", re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f​-‏"
                                 r"  ⁠﻿]"), " "),
    # Guiones o signos sueltos que quedan tras quitar marcas
    ("orphan_dashes", re.compile(r"(?:(?<=\s)|^)[-–—|]+(?=\s|$)"), " "),
]

_WS = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL = re.compile(r"\s*\n\s*")


def _normalize_unicode(text: str) -> str:
    """NFKC (convierte \\xa0 y compatibles), desescapa entidades HTML."""
    text = html.unescape(text)
    return unicodedata.normalize("NFKC", text)


def _collapse_ws(text: str) -> str:
    text = _MULTI_NL.sub("\n", text)
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


# Textos promocionales repetidos de sitios (21st Century Wire / ACR / Boiler Room): se quita la
# oración completa que contiene la marca.
_PROMO = re.compile(r"21st Century Wire|21WIRE|Alternate Current Radio|Boiler Room|\bACR\b|"
                    r"Reference Links|LISTEN TO THE SHOW|SUPPORT OUR WORK|Patreon", re.I)
_SENT_END = re.compile(r"[.!?\n]")

# Pre-chequeo barato (subcadena en minúscula) para no correr reglas caras en vano.
_GUARDS: dict[str, tuple[str, ...]] = {
    "html_tag": ("<",),
    "agency_dateline": ("(reuters", "(ap)", "(afp)"),
    "agency_correction": ("correct", "refile", "update", "add"),
    "reuters_template": ("following statements",),
    "reuters_template_tail": ("opinions expressed", "has not edited"),
    "sponsor_disclaimer": ("sap",),
    "daily_highlights": ("highlights of the day",),
    "agency_name": ("reuters",),
    "reuters_tweet_stamp": ("source link", "est]", "edt]"),
    "video_embed": ("video", "clip", "footage", "segment", "interview", "watch", "listen"),
    "pic_twitter": ("pic.twitter",),
    "tweet_signature": ("(@",),
    "url": ("http", "www."),
    "handle": ("@",),
    "featured_image": ("featured",),
    "photo_credit": ("photo", "image", "screen"),
    "getty": ("getty",),
    "via_line": (":",),
    "read_more": ("read more",),
}


def _remove_promo_sentences(text: str) -> str:
    """Quita cada oración que contiene una marca promocional de sitio."""
    while True:
        m = _PROMO.search(text)
        if not m:
            return text
        start = m.start()
        left = max((x.end() for x in _SENT_END.finditer(text, max(0, start - 300), start)),
                   default=max(0, start - 300) if start > 300 else 0)
        right_m = _SENT_END.search(text, m.end(), m.end() + 400)
        right = right_m.end() if right_m else min(len(text), m.end() + 400)
        text = text[:left] + " " + text[right:]


def _one_pass(text: str) -> str:
    low = text.lower()
    for name, pattern, repl in _RULES:
        guard = _GUARDS.get(name)
        if not guard or any(g in low for g in guard):
            text = pattern.sub(repl, text)
            low = text.lower()
        if name == "read_more":
            text = _remove_promo_sentences(text)
            low = text.lower()
    return _collapse_ws(text)


def clean_text(text: object, mode: str = "full") -> str:
    """Limpia un texto de noticia. Función pura e idempotente.

    Args:
        text: texto crudo (título y/o cuerpo). Valores no-str (None, NaN) se tratan como vacío.
        mode: ``"full"`` (modelo) o ``"minimal"`` (solo para el experimento de fuga).

    Returns:
        El texto limpio (puede ser vacío si no quedaba contenido).
    """
    if mode not in CLEAN_MODES:
        raise ValueError(f"Modo de limpieza desconocido: {mode!r}. Opciones: {CLEAN_MODES}")
    if not isinstance(text, str):
        return ""
    text = _collapse_ws(_normalize_unicode(text))
    if mode == "minimal":
        return text
    # Se itera hasta un punto fijo para garantizar idempotencia (normalmente 1-2 pasadas).
    for _ in range(4):
        new = _one_pass(text)
        if new == text:
            break
        text = new
    return text


def build_document(title: object, body: object, mode: str = "full") -> str:
    """Arma la entrada del modelo: título + salto de línea + cuerpo, y la limpia."""
    t = title if isinstance(title, str) else ""
    b = body if isinstance(body, str) else ""
    return clean_text(f"{t}\n{b}", mode=mode)


def rule_names() -> list[str]:
    """Nombres de las reglas de limpieza, en orden (para documentación y auditoría)."""
    names = [name for name, _, _ in _RULES]
    names.insert(names.index("read_more") + 1, "site_promo")
    return names


def rule_hit_counts(text: str) -> dict[str, int]:
    """Cuántas veces matchea cada regla sobre un texto (aplicando las anteriores primero)."""
    text = _collapse_ws(_normalize_unicode(text if isinstance(text, str) else ""))
    hits: dict[str, int] = {}
    for name, pattern, repl in _RULES:
        text, n = pattern.subn(repl, text)
        hits[name] = n
        if name == "read_more":
            n_promo = len(_PROMO.findall(text))
            text = _remove_promo_sentences(text)
            hits["site_promo"] = n_promo
    return hits


def word_count(text: str) -> int:
    """Cantidad de palabras (secuencias alfanuméricas)."""
    return len(re.findall(r"\b\w+\b", text or ""))
