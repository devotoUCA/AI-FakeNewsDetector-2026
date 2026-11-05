"""Fixtures comunes.

FIXTURE SINTÉTICO: los textos de `tiny_corpus` están escritos a mano SOLO para los tests
unitarios (forma, determinismo, contratos). No son datos reales ni se usan para métricas.
"""

from __future__ import annotations

import pytest

from fakenews.config import get_paths

REAL_STYLE = [
    "The finance ministry said on Tuesday that the budget deficit narrowed in the third quarter, "
    "according to official figures released by the statistics office.",
    "Lawmakers approved the bill after a lengthy debate, officials said, adding that the measure "
    "would take effect next year pending the president's signature.",
    "The central bank kept interest rates unchanged, a spokesman told reporters, citing stable "
    "inflation and moderate growth in the region.",
    "Officials confirmed that negotiations between the two governments will resume next month in "
    "Geneva, the foreign ministry said in a statement.",
    "The court ruled on Wednesday that the regulation was lawful, according to documents reviewed "
    "by the parties, and an appeal is expected.",
    "Election authorities said turnout reached 61 percent, slightly above the previous vote, and "
    "official results are expected on Friday.",
]
FAKE_STYLE = [
    "SHOCKING! You won't believe what this corrupt politician just did!!! The truth is finally "
    "exposed and the media is hiding it from you!",
    "BOMBSHELL: They don t want you to know this secret. Share before it gets deleted! Totally "
    "insane what is happening right now!!!",
    "WOW! This disgusting liar just got DESTROYED on live TV and everyone is furious! You need to "
    "see this before it is censored!",
    "BREAKING!!! The hoax is over. Evil traitors exposed in massive scandal, and you will be "
    "shocked by what we found!",
    "Unbelievable! The crazy truth about the corrupt elites is finally out and it is a total "
    "disaster for them! Must watch!!",
    "OMG! We told you so! This insane fraud is the biggest scandal ever and they are lying to you "
    "about everything!!!",
]


@pytest.fixture
def tiny_corpus() -> tuple[list[str], list[int]]:
    """Corpus sintético mínimo (6 reales, 6 falsas). Marcado como fixture de tests."""
    return REAL_STYLE + FAKE_STYLE, [0] * len(REAL_STYLE) + [1] * len(FAKE_STYLE)


def _have(*rel: str) -> bool:
    root = get_paths().root
    return all((root / r).exists() for r in rel)


needs_data = pytest.mark.skipif(
    not _have("data/processed/train.csv", "data/processed/val.csv", "data/processed/test.csv"),
    reason="Faltan los splits procesados (correr `make data`).")
needs_model = pytest.mark.skipif(
    not _have("models/final_model.joblib", "reports/val_results.json"),
    reason="Falta el modelo entrenado (correr `make train`).")
needs_eval = pytest.mark.skipif(
    not _have("reports/test_results.json"), reason="Falta la evaluación final (correr `make evaluate`).")
