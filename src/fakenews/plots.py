"""Figuras para el informe (PNG en `reports/figures/`). Estética sobria y consistente.

Colores: Real = azul, Falsa = naranja (paleta categórica validada para daltonismo); el resto
en grises. Una sola escala por gráfico, ejes y grilla recesivos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REAL = "#2a78d6"
FAKE = "#eb6834"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
NEUTRAL = "#8a8985"
SURFACE = "#fcfcfb"


def _style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def confusion_matrix_plot(cm: dict[str, int], title: str, path: Path) -> Path:
    """Matriz de confusión con conteos y porcentajes por fila (escala secuencial de un tono)."""
    m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    row = m / np.maximum(1, m.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]}\n({row[i, j]:.1%})", ha="center", va="center",
                    color="white" if row[i, j] > 0.6 else INK, fontsize=10)
    ax.set_xticks([0, 1], ["Real (pred.)", "Falsa (pred.)"])
    ax.set_yticks([0, 1], ["Real", "Falsa"])
    _style(ax, title)
    ax.grid(False)
    return _save(fig, path)


def reliability_plot(curves: dict[str, list[dict[str, float]]], title: str, path: Path) -> Path:
    """Diagrama de confiabilidad: probabilidad media vs. fracción real de falsas por bin."""
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=1, linestyle="--", label="Calibración perfecta")
    for (name, rows), color in zip(curves.items(), SERIES):
        pts = [(r["p_media"], r["frac_falsas"]) for r in rows if r["n"] > 0]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=2, color=color, label=name)
    _style(ax, title, "Probabilidad de falsa predicha", "Fracción real de falsas")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


def pr_curve_plot(pr: dict[str, list[float]], chosen: dict[str, float], path: Path) -> Path:
    """Curva precisión-recall de la clase falsa en validación, con el umbral elegido."""
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot(pr["recall"], pr["precision"], color=FAKE, linewidth=2, label="Validación")
    ax.scatter([chosen["recall_fake"]], [chosen["precision_fake"]], s=50, color=INK, zorder=3,
               label=f"Umbral elegido = {chosen['umbral']:.2f}")
    _style(ax, "Curva precisión–recall (clase falsa)", "Recall", "Precisión")
    lo = min(0.9, min(pr["precision"]))
    ax.set_ylim(lo, 1.002)
    ax.set_xlim(lo, 1.002)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    return _save(fig, path)


def coverage_plot(rows: list[dict[str, float]], path: Path, title: str) -> Path:
    """Cobertura (fracción no derivada) vs. accuracy de los casos cubiertos."""
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    xs = [r["cobertura"] for r in rows]
    ys = [r["accuracy_cubiertos"] for r in rows]
    ax.plot(xs, ys, marker="o", markersize=4, linewidth=2, color=REAL)
    for r in rows:
        if abs(r["semiancho"] - 0.10) < 1e-9:
            ax.annotate("zona 0,4–0,6", (r["cobertura"], r["accuracy_cubiertos"]), fontsize=8,
                        color=INK2, xytext=(-70, 8), textcoords="offset points")
    _style(ax, title, "Cobertura (fracción que el sistema decide)", "Accuracy de lo decidido")
    ax.invert_xaxis()
    return _save(fig, path)


def bar_compare(labels: list[str], values: dict[str, list[float]], title: str, ylabel: str,
                path: Path, ylim: tuple[float, float] | None = None,
                colors: list[str] | None = None) -> Path:
    """Barras agrupadas (una serie por métrica). `colors` permite fijar el color por entidad."""
    fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * len(labels) + 2), 4.0))
    n = len(values)
    width = 0.8 / n
    x = np.arange(len(labels))
    for i, ((name, vals), color) in enumerate(zip(values.items(), colors or SERIES)):
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width * 0.9, color=color, label=name)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    _style(ax, title, "", ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    if n > 1:
        ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


def line_plot(series: dict[str, tuple[list[float], list[float]]], title: str, xlabel: str,
              ylabel: str, path: Path, logy: bool = False) -> Path:
    """Varias series sobre un mismo eje (p. ej. pérdida por época)."""
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for (name, (xs, ys)), color in zip(series.items(), SERIES):
        ax.plot(xs, ys, linewidth=2, marker="o", markersize=3, color=color, label=name)
    _style(ax, title, xlabel, ylabel)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


def hist_by_class(values: dict[str, np.ndarray], title: str, xlabel: str, path: Path,
                  bins: int = 50, log: bool = False) -> Path:
    """Histogramas superpuestos por clase."""
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for (name, vals), color in zip(values.items(), [REAL, FAKE, NEUTRAL]):
        ax.hist(vals, bins=bins, alpha=0.55, color=color, label=name, log=log)
    _style(ax, title, xlabel, "Cantidad")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


def stacked_share(categories: list[str], fake_share: list[float], counts: list[int], title: str,
                  path: Path) -> Path:
    """Proporción de falsas por grupo (barra horizontal)."""
    fig, ax = plt.subplots(figsize=(5.6, 0.45 * len(categories) + 1.2))
    y = np.arange(len(categories))
    ax.barh(y, fake_share, color=FAKE, height=0.6, label="Falsas")
    ax.barh(y, [1 - s for s in fake_share], left=fake_share, color=REAL, height=0.6, label="Reales")
    for yi, c in zip(y, counts):
        ax.text(1.01, yi, f"n={c}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, categories, fontsize=8)
    _style(ax, title, "Proporción", "")
    ax.set_xlim(0, 1.12)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    return _save(fig, path)


def save_all_notice(paths: list[Any]) -> list[str]:
    """Devuelve rutas como texto (para registrar en los reportes)."""
    return [str(p) for p in paths]
