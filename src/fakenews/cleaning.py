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


def _one_pass(text: str, skip: frozenset[str] = frozenset()) -> str:
    low = text.lower()
    for name, pattern, repl in _RULES:
        if name in skip:
            continue
        guard = _GUARDS.get(name)
        if not guard or any(g in low for g in guard):
            text = pattern.sub(repl, text)
            low = text.lower()
        if name == "read_more":
            text = _remove_promo_sentences(text)
            low = text.lower()
    return _collapse_ws(text)


# --- Perfiles v2 / v3 (iteración 2): "neutral a la fuente" ---------------------------------
# v2 reemplaza el tratamiento de comillas, URLs y handles de v1 por una normalización explícita.
_V2_SKIP = frozenset({"url", "domain", "handle", "quotes"})
_V2_STEPS: list[tuple[str, re.Pattern[str], str]] = [
    ("curly_single", re.compile(r"[‘’‚‛`´]"), "'"),
    ("curly_double", re.compile(r"[“”„«»]"), '"'),
    # Contracciones: "don't", "don’t" y "don t" (apóstrofo perdido) colapsan a la misma forma.
    ("ctr_cant", re.compile(r"\bcan(?:'|\s)t\b", re.I), "cannot"),
    ("ctr_wont", re.compile(r"\bwon(?:'|\s)t\b", re.I), "will not"),
    ("ctr_aint", re.compile(r"\bain(?:'|\s)t\b", re.I), "is not"),
    ("ctr_nt_space", re.compile(r"\b(do|does|did|is|are|was|were|has|have|had|could|would|should|"
                                r"must|need|might)n(?:'|\s)t\b", re.I), r"\1 not"),
    ("ctr_nt", re.compile(r"\b(\w+)n't\b", re.I), r"\1 not"),
    ("ctr_ll", re.compile(r"\b(i|you|we|they|he|she|it|that|there|who|what)(?:'|\s)ll\b", re.I),
     r"\1 will"),
    ("ctr_re", re.compile(r"\b(you|we|they|who|what)(?:'|\s)re\b", re.I), r"\1 are"),
    ("ctr_ve", re.compile(r"\b(i|you|we|they|who|could|would|should|might|must)(?:'|\s)ve\b",
                          re.I), r"\1 have"),
    ("ctr_m", re.compile(r"\b(i)(?:'|\s)m\b", re.I), r"\1 am"),
    ("ctr_d", re.compile(r"\b(i|you|he|she|we|they|it|that|who)(?:'|\s)d\b", re.I), r"\1 would"),
    ("ctr_s_is", re.compile(r"\b(it|that|there|here|what|who|he|she|where|how)(?:'|\s)s\b", re.I),
     r"\1 is"),
    ("possessive", re.compile(r"(?<=[A-Za-z])(?:'s\b|'(?=\s)|\s+s\b(?!\.))"), ""),
    ("other_quotes", re.compile(r"[\"']"), " "),
    # Enlaces, usuarios y hashtags a tokens neutros (mismo token sin importar el sitio).
    ("url_token", re.compile(r"(?:https?://|\bwww\.)\S*", re.I), " xurlx "),
    ("domain_token", re.compile(r"\b[\w\-]+(?:\.[\w\-]+)*\.(?:com|org|net|co|ly|gov|edu|tv|info|us)"
                                r"(?:/\S*)?\b", re.I), " xurlx "),
    ("handle_token", re.compile(r"@\w{1,30}"), " xuserx "),
    ("hashtag_token", re.compile(r"#\w{1,60}"), " xhashtagx "),
    # Siglas con puntos (estilo de agencia "U.S.") a la forma sin puntos.
    ("acronym_dots", re.compile(r"\b(?:U\.S\.A\.|U\.S\.|U\.K\.|U\.N\.|E\.U\.)"),
     "__ACR__"),
    # Horas y husos horarios (estilo de agencia "10 a.m. EDT (1400 GMT)").
    ("clock_time", re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:[ap]\.m\.|[ap]m\b)", re.I), " xtimex "),
    ("time_zone", re.compile(r"\b(?:GMT|EDT|EST|PDT|PST|CDT|CST|UTC|ET)\b"), " "),
    # Días y meses fuera (las reales nombran el día de la semana casi siempre).
    ("weekday", re.compile(r"\b(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)days?\b", re.I), " "),
    ("month", re.compile(r"\b(?:January|February|March|April|June|July|August|September|October|"
                         r"November|December|JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|"
                         r"OCTOBER|NOVEMBER|DECEMBER)\b|\bMay(?=\s+\d)|\b(?:Jan|Feb|Mar|Apr|Aug|Sept?|"
                         r"Oct|Nov|Dec)\.(?=\s*\d)"), " "),
    # Números (y ordinales) a token.
    ("number_token", re.compile(r"(?<![A-Za-z])\d+(?:[.,:]\d+)*(?:st|nd|rd|th)?(?![A-Za-z\d])"), " xnumx "),
]
_ACR_MAP = {"U.S.A.": "USA", "U.S.": "US", "U.K.": "UK", "U.N.": "UN", "E.U.": "EU"}
_ACR_RE = re.compile(r"\b(?:U\.S\.A\.|U\.S\.|U\.K\.|U\.N\.|E\.U\.)")
# v3: además se enmascara vocabulario de redacción de agencia (lista en config.yaml,
# justificada por reports/shortcut_audit_v2.csv), se quitan los tokens de enlaces, se pasan a
# minúscula las palabras TODO MAYÚSCULAS de 5+ letras y se colapsan signos repetidos.
_CAPS_WORD = re.compile(r"\b[A-Z]{5,}\b")
_REPEAT_PUNCT = re.compile(r"([!?])\1+")
# La auditoría mostró que los tokens neutros de enlaces siguen delatando la fuente (AUC 0,63):
# v3 los quita.
_LINK_TOKENS = re.compile(r"\bx(?:url|user|hashtag)x\b")


def _v2_pass(text: str) -> str:
    text = _one_pass(text, skip=_V2_SKIP)
    text = _ACR_RE.sub(lambda m: _ACR_MAP[m.group(0)], text)
    for name, pattern, repl in _V2_STEPS:
        if name == "acronym_dots":
            continue
        text = pattern.sub(repl, text)
    return _collapse_ws(text)


_MASK_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}


def _mask_pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    if terms not in _MASK_CACHE:
        alts = "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in sorted(terms, key=len, reverse=True))
        _MASK_CACHE[terms] = re.compile(rf"\b(?:{alts})\b", re.I)
    return _MASK_CACHE[terms]


def _v3_pass(text: str, mask_terms: tuple[str, ...]) -> str:
    text = _v2_pass(text)
    if mask_terms:
        text = _mask_pattern(mask_terms).sub(" ", text)
    text = _LINK_TOKENS.sub(" ", text)
    text = _CAPS_WORD.sub(lambda m: m.group(0).lower(), text)
    text = _REPEAT_PUNCT.sub(r"\1", text)
    return _collapse_ws(text)


# --- Perfil v2_1 (iteración 3): v2 + normalización de espacios y puntuación ------------------
# Justificado por reports/format_shortcut_discovery_v3.csv: la "oración pegada" (`a.A`) predice la
# etiqueta sola con AUC ~0,94 aun después de v2, y los espacios antes de puntuación son huecos que
# deja la propia limpieza. `v2` no se modifica: los resultados ya reportados siguen reproducibles.
_V21_STEPS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\.{2,}|…"), "."),                                    # puntos suspensivos
    (re.compile(r"[—–−]"), "-"),                                        # guiones largos y medios
    (re.compile(r"[ \t]+([.,;:!?)\]])"), r"\1"),                        # espacio antes de puntuación
    (re.compile(r"([(\[])[ \t]+"), r"\1"),                              # espacio tras apertura
    (re.compile(r"\(\s*\)|\[\s*\]"), " "),                              # paréntesis vacíos
    (re.compile(r"([.,;:])(?:\s*[.,;:])+"), r"\1"),                     # puntuación repetida
    (re.compile(r"(?<=[A-Za-z)\]])([.!?;:,])(?=[A-Za-z(])"), r"\1 "),   # oración pegada
    (re.compile(r"(?m)^[ \t.,;:]+"), ""),                               # puntuación huérfana inicial
]


def _v2_1_pass(text: str) -> str:
    text = _v2_pass(text)
    for pattern, repl in _V21_STEPS:
        text = pattern.sub(repl, text)
    return _collapse_ws(text)


PROFILES = ("v1", "v2", "v2_1", "v3")


def v3_mask_terms() -> tuple[str, ...]:
    """Términos que v3 enmascara (de config.yaml, sección cleaning.v3_mask_terms)."""
    from .config import load_config

    return tuple(load_config().get("cleaning", {}).get("v3_mask_terms", []))


def clean_text(text: object, mode: str = "full", profile: str = "v1") -> str:
    """Limpia un texto de noticia. Función pura e idempotente.

    Args:
        text: texto crudo (título y/o cuerpo). Valores no-str (None, NaN) se tratan como vacío.
        mode: ``"full"`` (modelo) o ``"minimal"`` (solo para el experimento de fuga).
        profile: perfil de limpieza para ``mode="full"``: ``"v1"`` (iteración 1), ``"v2"``
            (neutral a la fuente: contracciones, tokens neutros, sin días/meses/números) o
            ``"v2_1"`` (v2 + espacios y puntuación normalizados) o ``"v3"`` (v2 + enmascarado de
            vocabulario de agencia y de mayúsculas).

    Returns:
        El texto limpio (puede ser vacío si no quedaba contenido).
    """
    if mode not in CLEAN_MODES:
        raise ValueError(f"Modo de limpieza desconocido: {mode!r}. Opciones: {CLEAN_MODES}")
    if profile not in PROFILES:
        raise ValueError(f"Perfil de limpieza desconocido: {profile!r}. Opciones: {PROFILES}")
    if not isinstance(text, str):
        return ""
    text = _collapse_ws(_normalize_unicode(text))
    if mode == "minimal":
        return text
    if profile == "v1":
        step = _one_pass
    elif profile == "v2":
        step = _v2_pass
    elif profile == "v2_1":
        step = _v2_1_pass
    else:
        terms = v3_mask_terms()

        def step(t: str) -> str:
            return _v3_pass(t, terms)
    # Se itera hasta un punto fijo para garantizar idempotencia (normalmente 1-2 pasadas).
    for _ in range(5):
        new = step(text)
        if new == text:
            break
        text = new
    return text


def build_document(title: object, body: object, mode: str = "full", profile: str = "v1") -> str:
    """Arma la entrada del modelo: título + salto de línea + cuerpo, y la limpia."""
    t = title if isinstance(title, str) else ""
    b = body if isinstance(body, str) else ""
    return clean_text(f"{t}\n{b}", mode=mode, profile=profile)


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
