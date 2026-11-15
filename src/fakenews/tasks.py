"""Runner multiplataforma: `python -m fakenews.tasks <tarea>` (reemplaza a `make` en Windows).

Solo usa la biblioteca estándar, para poder correr `setup` antes de instalar nada:

    PYTHONPATH=src python -m fakenews.tasks setup        # macOS / Linux
    $env:PYTHONPATH="src"; python -m fakenews.tasks setup  # Windows PowerShell

El resto de las tareas usa el entorno `.venv` si existe y, si no, el intérprete actual.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENV = ROOT / ".venv"


def venv_python() -> str:
    """Python del entorno del proyecto (funciona en Windows y en Unix)."""
    for candidate in (VENV / "Scripts" / "python.exe", VENV / "bin" / "python"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def cli(*args: str) -> list[str]:
    return [venv_python(), "-m", "fakenews.cli", *args]


# Cada tarea es una lista de comandos; el Makefile es un envoltorio fino de este diccionario.
TASKS: dict[str, list[list[str]]] = {
    "data": [cli("download"), cli("prepare"), cli("audit"), cli("download-external"), cli("prepare-external")],
    "train": [cli("train"), cli("shortcuts"), cli("panel"), cli("train-v2"), cli("format-shortcuts"),
              cli("panel-v3"), cli("train-v3")],
    "quickstart": [cli("quickstart")],
    "evaluate": [cli("evaluate"), cli("evaluate-v2"), cli("evaluate-v3")],
    "analyze": [cli("analyze")],
    "review": [cli("review-status")],
    "report": [cli("report")],
    "notebook": [[venv_python(), "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                  "notebooks/01_eda_y_preparacion.ipynb", "--ExecutePreprocessor.timeout=1200"]],
    "test": [[venv_python(), "-m", "pytest", "-q"]],
    "test-fast": [[venv_python(), "-m", "pytest", "-q", "-m", "not slow"]],
    "demo": [[venv_python(), "-m", "streamlit", "run", "app/demo.py"]],
}
TASKS["all"] = [c for k in ("data", "train", "evaluate", "analyze", "report", "notebook", "test") for c in TASKS[k]]


def setup() -> int:
    """Crea `.venv` e instala dependencias fijadas y el paquete."""
    if not VENV.exists():
        venv.create(VENV, with_pip=True)
    py = venv_python()
    for cmd in ([py, "-m", "pip", "install", "--upgrade", "pip"], [py, "-m", "pip", "install", "-r", "requirements.txt"],
                [py, "-m", "pip", "install", "-e", ".", "--no-deps"]):
        code = subprocess.call(cmd, cwd=ROOT)
        if code:
            return code
    return 0


def run(name: str, dry_run: bool = False) -> int:
    """Ejecuta una tarea. Con `dry_run` solo imprime los comandos."""
    if name == "setup":
        if dry_run:
            print("setup: crea .venv e instala requirements.txt y el paquete")
            return 0
        return setup()
    for cmd in TASKS[name]:
        print("$", " ".join(cmd), flush=True)
        if dry_run:
            continue
        code = subprocess.call(cmd, cwd=ROOT, env=os.environ.copy())
        if code:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fakenews.tasks", description="Runner multiplataforma del proyecto")
    parser.add_argument("task", choices=["setup", *TASKS])
    parser.add_argument("--dry-run", action="store_true", help="Mostrar los comandos sin ejecutarlos")
    args = parser.parse_args(argv)
    return run(args.task, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
