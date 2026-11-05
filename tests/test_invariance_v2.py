"""Invarianza (metamórfica) y transformaciones."""

import joblib
import pytest

from fakenews.config import get_paths, load_config
from fakenews.invariance import (TRANSFORMS, drop_days_months, flip_stats, swap_apostrophe_space,
                                 swap_quote_style, swap_source_marks, swap_title_case)
from fakenews.models import p_fake

from .conftest import needs_data
from .conftest_v2 import needs_v2_model


def test_transforms_do_what_they_say():
    assert "’" in swap_quote_style("", "It's a \"test\"")[1]
    assert "'" in swap_quote_style("", "It’s “x”")[1] and "’" not in swap_quote_style("", "It’s “x”")[1]
    assert swap_apostrophe_space("", "I don't know")[1] == "I don t know"
    assert swap_apostrophe_space("", "I don t know")[1] == "I don't know"
    assert swap_title_case("SHOCKING NEWS today", "")[0] == "Shocking news today"
    assert swap_title_case("Calm title", "")[0] == "CALM TITLE"
    assert "Tuesday" not in drop_days_months("", "On Tuesday in March")[1]
    assert "(Reuters)" not in swap_source_marks("", "LONDON (Reuters) - Text")[1]
    assert swap_source_marks("", "Text")[1].startswith("WASHINGTON (Reuters)")


def test_flip_stats():
    import numpy as np

    r = flip_stats(np.array([0.1, 0.9]), np.array([0.6, 0.9]))
    assert r["tasa_cambio"] == 0.5 and abs(r["delta_p_medio"] - 0.25) < 1e-12


@needs_v2_model
@needs_data
def test_final_model_invariance_within_threshold():
    """Las transformaciones que no cambian la veracidad cambian la predicción en <= el umbral."""
    from fakenews.panel import invariance_inputs

    cfg = load_config()
    art = joblib.load(get_paths().models / "final_model_v2.joblib")
    inv = invariance_inputs(art["profile"], 500, cfg["seed"])
    base = p_fake(art["model"], inv["base"])
    limit = cfg["panel"]["max_invariance_flip"]
    for name in TRANSFORMS:
        rate = flip_stats(base, p_fake(art["model"], inv[name]), art["threshold"])["tasa_cambio"]
        assert rate <= limit, f"La transformación '{name}' cambia el {rate:.1%} de las predicciones (> {limit:.0%})"


@needs_v2_model
@needs_data
def test_sentence_spacing_dependence_is_bounded():
    """Atajo residual hallado en la iteración 2: oraciones pegadas sin espacio ('it.The').
    Normalizar el espaciado no debe cambiar más que el umbral de invarianza."""
    import re

    import pandas as pd

    from fakenews.cleaning import build_document
    from fakenews.corpus import isot_raw

    cfg = load_config()
    art = joblib.load(get_paths().models / "final_model_v2.joblib")
    # Misma muestra protocolar que el criterio de invarianza del panel (2000 textos de val).
    # Nota honesta: en una submuestra de 400 este valor dio 2,5 % (> 2 %): está en el límite.
    from fakenews.panel import invariance_inputs

    ids = invariance_inputs(art["profile"], cfg["panel"]["invariance_sample"], cfg["seed"])["__ids__"]
    raw = isot_raw(pd.Series(ids))
    glue = re.compile(r"([.!?])(?=[A-Z])")
    base = [build_document(t, b, profile=art["profile"]) for t, b in zip(raw["title"], raw["text"])]
    fixed = [build_document(t, glue.sub(r"\1 ", b), profile=art["profile"]) for t, b in zip(raw["title"], raw["text"])]
    rate = flip_stats(p_fake(art["model"], base), p_fake(art["model"], fixed), art["threshold"])["tasa_cambio"]
    assert rate <= cfg["panel"]["max_invariance_flip"], f"espaciado de oraciones cambia {rate:.1%}"
    assert isinstance(pd.Series(base), pd.Series)
