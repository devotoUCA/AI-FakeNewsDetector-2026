"""Log de accesos a conjuntos reservados: entradas retroactivas y verificación (punto A3).

`reports/test_access_log.jsonl` es append-only. Lo que no quedó registrado a tiempo se agrega con
`"retroactive": true`, con la fecha real de registro en `ts` y la fuente de la información: nunca se
falsifican fechas. `verify_access_log` comprueba que cada cifra de holdout citada en los documentos
exista en los reportes y que su conjunto e iteración tengan entrada en el log.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from .config import get_paths, read_json

DOCS = ("README.md", "docs/informe_resultados.md", "docs/model_card.md", "docs/guion_defensa.md",
        "docs/campos_completar.md", "docs/02_iteracion_3.md")
# Palabras que identifican una línea que cita un conjunto reservado.
HOLDOUT_WORDS = {"isot_test": ("ISOT test",), "welfake_holdout": ("WELFake holdout", "WELFake-holdout"),
                 "fakes_holdout": ("FA-KES",), "liar_test": ("LIAR test",)}
_DEC = re.compile(r"(?<![\d,.])0,\d{2,4}(?![\d])")
RETRO = [
    {"modelo": m, "conjunto": "liar_train+valid+test", "iteracion": 1, "reacceso": False,
     "fecha_evento": "2026-11-04", "fuente": "src/fakenews/evaluate.py (evaluate_external usaba load_liar completo) "
     "y reports/external_liar.json; docs/01_iteracion_2.md lo declara",
     "motivo": "validación externa de la iteración 1, no registrada en su momento"}
    for m in ("dummy", "logreg_raw", "linear_svm_cal", "style_forest", "knn", "mlp")
] + [
    {"modelo": "todos los modelos de la iteración 1 (copia limpia de verificación)", "conjunto": c, "iteracion": 1,
     "reacceso": True, "fecha_evento": "2026-11-04", "fuente": "verificación de `make all` en una copia limpia "
     "durante la iteración 1: su log quedó en esa copia; dio cifras idénticas y no se tomó ninguna decisión",
     "motivo": "chequeo de reproducibilidad"}
    for c in ("isot_test", "liar_train+valid+test")
]


def log_path() -> Path:
    return get_paths().reports / "test_access_log.jsonl"


def read_log() -> list[dict[str, Any]]:
    path = log_path()
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x] if path.exists() else []


def add_retroactive_entries() -> int:
    """Agrega las entradas retroactivas faltantes (idempotente). Devuelve cuántas agregó."""
    log = read_log()
    have = {(e["modelo"], e.get("conjunto")) for e in log if e.get("retroactive")}
    now = dt.datetime.now().isoformat(timespec="seconds")
    new = [{"ts": now, "retroactive": True, "huella": "desconocida", **e} for e in RETRO
           if (e["modelo"], e["conjunto"]) not in have]
    with log_path().open("a", encoding="utf-8") as fh:
        for e in new:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    return len(new)


def _variants(value: float) -> set[str]:
    return {f"{value:.{n}f}".replace(".", ",") for n in (2, 3, 4)}


def verify_access_log(root: Path | None = None) -> dict[str, Any]:
    """Comprueba log y cifras citadas. Devuelve {"ok": bool, "problemas": [...]}."""
    paths = get_paths()
    root = root or paths.root
    problems: list[str] = []
    registry = read_json(paths.reports / "holdout_figures_v3.json")
    log = read_log()
    logged = {(e.get("conjunto", "isot_test").split(" ")[0], e.get("iteracion", 1)) for e in log}
    logged |= {("liar_test", e.get("iteracion", 1)) for e in log if "liar" in e.get("conjunto", "")}
    for item in registry["cifras"]:
        if (item["conjunto"], item["iteracion"]) not in logged:
            problems.append(f"sin entrada en el log: {item['conjunto']} iteración {item['iteracion']}")
    allowed: dict[str, set[str]] = {k: set() for k in HOLDOUT_WORDS}
    for item in registry["cifras"]:
        allowed[item["conjunto"]] |= _variants(item["valor"])
    checked = 0
    for rel in DOCS:
        path = root / rel
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            hits = [k for k, ws in HOLDOUT_WORDS.items() if any(w in line for w in ws)]
            if not hits or "no-verificar" in line:
                continue
            pool = set().union(*(allowed[k] for k in hits))
            for num in _DEC.findall(line):
                checked += 1
                if num not in pool:
                    problems.append(f"{rel}:{i}: la cifra {num} no sale de los reportes de {hits}")
    return {"ok": not problems, "cifras_revisadas": checked, "problemas": sorted(set(problems))}


if __name__ == "__main__":
    import sys

    if "--add-retroactive" in sys.argv:
        print("entradas agregadas:", add_retroactive_entries())
    r = verify_access_log()
    print(json.dumps(r, ensure_ascii=False, indent=2))
    sys.exit(0 if r["ok"] else 1)
