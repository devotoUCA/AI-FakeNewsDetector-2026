"""Panel: folds por grupo/tiempo sin solapamiento; el test no se toca durante la selección."""

import json

import numpy as np
import pandas as pd

from fakenews.config import get_paths, read_json
from fakenews.corpus import isot_split
from fakenews.panel import group_folds, select_robust, temporal_split

from .conftest import needs_data
from .conftest_v2 import needs_panel


@needs_data
def test_group_and_temporal_folds_disjoint_and_two_classes():
    pool = pd.concat([isot_split("train", "v1"), isot_split("val", "v1")], ignore_index=True)
    test_ids = set(isot_split("test", "v1")["id"])
    assert not (set(pool["id"]) & test_ids)
    folds = group_folds(pool)
    held_subjects = set()
    for f in folds:
        tr, ev = set(f["train_idx"]), set(f["eval_idx"])
        assert not tr & ev
        ev_df = pool.iloc[f["eval_idx"]]
        assert ev_df["label"].nunique() == 2
        assert not set(pool.iloc[f["train_idx"]]["subject"]) & set(ev_df["subject"])
        held_subjects |= set(ev_df["subject"])
    assert held_subjects == set(pool["subject"])  # cada subject queda afuera al menos una vez
    ts = temporal_split(pool)
    dt = pd.to_datetime(pool["date"].str.strip(), errors="coerce", format="mixed")
    assert dt.iloc[ts["train_idx"]].max() < dt.iloc[ts["eval_idx"]].min()
    assert not set(ts["train_idx"]) & set(ts["eval_idx"])


def test_select_robust_rule():
    def r(name, model, score, ok=True):
        return {"candidato": name, "spec": {"model": model}, "puntaje_robusto": score, "elegible": ok}
    res = [r("a", "ensemble", 0.80), r("b", "lr_word", 0.798), r("c", "lr_word", 0.90, ok=False)]
    sel = select_robust(res)
    assert sel["elegido"] == "b"  # empate (<0,005) con el ensamble -> el más simple; c no es elegible


@needs_panel
def test_selection_did_not_touch_test():
    """La selección del panel es ANTERIOR a cualquier acceso de la iteración 2 al test/holdout,
    y el panel no contiene métricas de test."""
    import datetime as dt
    import os

    paths = get_paths()
    panel = paths.reports / "panel_v2.json"
    out = read_json(panel)
    generated = out.get("generado") or dt.datetime.fromtimestamp(os.path.getmtime(panel)).isoformat(timespec="seconds")
    log = paths.reports / "test_access_log.jsonl"
    entries = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x]
    it2 = [e["ts"] for e in entries if e.get("iteracion") == 2]
    if it2:
        assert min(it2) >= generated, "hubo un acceso de la iteración 2 al test antes de cerrar la selección"
    names = {r["candidato"] for r in out["resultados"]}
    assert out["seleccion"]["elegido"] in names
    for r in out["resultados"]:
        assert set(r) >= {"in_domain", "grupo", "tiempo", "externo_dev_welfake", "invarianza"}
        assert "isot_test" not in r and "welfake_holdout" not in r
    assert np.isfinite(out["resultados"][0]["puntaje_robusto"])
