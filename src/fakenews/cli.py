"""Línea de comandos: `python -m fakenews.cli <comando>`.

Comandos: download, prepare, audit, train, evaluate, external, analyze, predict, all.
Lista completa (iteraciones 2 y 3, `report`, `review-status`): `python -m fakenews.cli --help`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import setup_logging

LOGGER = logging.getLogger("fakenews.cli")


def _force_utf8_output() -> None:
    """En Windows la consola suele usar cp1252: sin esto, imprimir "≥" o "–" puede fallar."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _read_stdin_text() -> str:
    """Lee stdin como UTF-8 (los ejemplos lo son), sin depender de la codificación de la consola."""
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is None:
        return sys.stdin.read()
    return buffer.read().decode("utf-8-sig", errors="replace")


def _cmd_download(_: argparse.Namespace) -> None:
    from .data import download_datasets

    print(json.dumps(download_datasets(), ensure_ascii=False, indent=2))


def _cmd_prepare(_: argparse.Namespace) -> None:
    from .splits import prepare_dataset

    m = prepare_dataset()
    print(json.dumps({k: v["total"] for k, v in m["splits"].items()} | {"sin_solapamiento": m["solapamiento"]["ok"]}))


def _cmd_audit(_: argparse.Namespace) -> None:
    from .audit import run_audit

    run_audit()
    LOGGER.info("Auditoría guardada en reports/data_audit.json y reports/leakage_audit.json")


def _cmd_train(_: argparse.Namespace) -> None:
    from .train import train_all

    r = train_all()
    print(json.dumps({"productivo": r["productivo"]["modelo"], "umbral": r["decision"]["umbral"]}))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from .evaluate import evaluate_external, evaluate_final, success_criteria

    r = evaluate_final(reason=args.reason)
    evaluate_external()
    s = success_criteria()
    m = r["modelos"][r["productivo"]]
    print(json.dumps({"productivo": r["productivo"], "accuracy_test": round(m["accuracy"], 4),
                      "f1_falsa_test": round(m["fake"]["f1"], 4), "f1_real_test": round(m["real"]["f1"], 4)}))
    for c in s["criterios"]:
        print(f"- {c['criterio']}: {c['estado']}")


def _cmd_external(_: argparse.Namespace) -> None:
    from .evaluate import evaluate_external

    evaluate_external()


def _cmd_analyze(_: argparse.Namespace) -> None:
    from .analyze import clustering_analysis, genetic_feature_selection

    clustering_analysis()
    genetic_feature_selection()


def _cmd_predict(args: argparse.Namespace) -> None:
    from .predict import ModelNotTrainedError, predict

    text = args.text if args.text != "-" else _read_stdin_text()
    try:
        out = predict(text, with_similar=args.similar)
    except ModelNotTrainedError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def _cmd_download_external(_: argparse.Namespace) -> None:
    from .data import download_external

    print(json.dumps(download_external(), ensure_ascii=False, indent=2))


def _cmd_prepare_external(_: argparse.Namespace) -> None:
    from .corpus import prepare_fakes, prepare_welfake

    m = prepare_welfake()
    f = prepare_fakes()
    print(json.dumps({"fakes": f["n"], "falsas": f["falsas"]}))
    print(json.dumps({k: v["n"] for k, v in m["splits"].items()} | {"etiquetas": m["direccion_etiquetas"]["direccion"]},
                     ensure_ascii=False))


def _cmd_shortcuts(_: argparse.Namespace) -> None:
    from .shortcuts import run_shortcut_audit

    run_shortcut_audit()
    LOGGER.info("Auditoría de atajos en reports/shortcut_audit_v2.csv")


def _cmd_panel(args: argparse.Namespace) -> None:
    from .panel import run_panel

    out = run_panel(only=args.only or None, n_jobs=args.jobs)
    print(json.dumps(out["seleccion"], ensure_ascii=False, indent=2))


def _cmd_train_v2(args: argparse.Namespace) -> None:
    from .train_v2 import train_final_v2

    out = train_final_v2(quick=getattr(args, "quick", False))
    print(json.dumps({"spec": out.get("spec", {}).get("name") if "spec" in out else None,
                      "metricas_val": out.get("metricas_val")}, ensure_ascii=False, default=str))


def _cmd_evaluate_v2(_: argparse.Namespace) -> None:
    from .evaluate_v2 import evaluate_final_v2, success_criteria_v2

    r = evaluate_final_v2()
    s = success_criteria_v2()
    print(json.dumps({"isot_test": r["v2"]["isot_test"]["forzado"]["accuracy"],
                      "welfake_holdout": r["v2"]["welfake_holdout"]["forzado"]["accuracy"],
                      "welfake_trivial": r["v2"]["welfake_holdout"]["trivial"]["accuracy"]}, ensure_ascii=False))
    for c in s["criterios"]:
        print(f"- {c['criterio']}: {c['estado']}")


def _cmd_format_shortcuts(_: argparse.Namespace) -> None:
    from .format_shortcuts import run_format_discovery

    run_format_discovery(("v2", "v2_1"))


def _cmd_panel_v3(args: argparse.Namespace) -> None:
    from .panel import run_panel_v3

    print(json.dumps(run_panel_v3(only=getattr(args, "only", None) or None, n_jobs=getattr(args, "jobs", 2))["seleccion"],
                     ensure_ascii=False, indent=2, default=str))


def _cmd_train_v3(args: argparse.Namespace) -> None:
    from .train_v3 import train_final_v3

    out = train_final_v3(quick=getattr(args, "quick", False))
    print(json.dumps({"modelo": out.get("spec", {}).get("name"), "dominio_dev": out.get("dominio_dev")}, ensure_ascii=False, default=str))


def _cmd_evaluate_v3(_: argparse.Namespace) -> None:
    from .evaluate_v3 import evaluate_final_v3, success_criteria_v3

    r = evaluate_final_v3()
    s = success_criteria_v3()
    print(json.dumps({k: round(v["accuracy"], 4) for k, v in r["v3"].items()}, ensure_ascii=False))
    for c in s["criterios"]:
        print(f"- {c['criterio']}: {c['estado']}")


def _cmd_report(_: argparse.Namespace) -> None:
    from .access_log import verify_access_log
    from .make_report import make_report
    from .review import ReviewFileError, update_review_status

    # Primero se actualiza el criterio de revisión de errores, así las tablas salen con el estado real.
    try:
        print(json.dumps({"revision_errores": update_review_status()}, ensure_ascii=False, indent=2))
    except ReviewFileError as exc:
        print(f"Revisión de errores: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(make_report(), ensure_ascii=False, indent=2))
    r = verify_access_log()
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if not r["ok"]:
        sys.exit(1)


def _cmd_review_status(_: argparse.Namespace) -> None:
    """Estado de la revisión humana de errores. No carga modelos ni datos reservados."""
    from .review import ReviewFileError, update_review_status

    try:
        s = update_review_status()
    except ReviewFileError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(json.dumps(s, ensure_ascii=False, indent=2))
    falta = max(0, s["minimo"] - s["revisados"])
    print(f"Revisados (comentario + firma): {s['revisados']} de {s['casos']}. "
          + ("Criterio del acta cumplido." if s["cumple"] else f"Faltan {falta} para cumplir el criterio del acta."))
    if s["comentados_sin_firma"]:
        print(f"Hay {s['comentados_sin_firma']} comentarios sin firma en `revisado_por`: no cuentan todavía.")
    print("Para llevar el estado a los documentos: python -m fakenews.cli report")


def _cmd_quickstart(args: argparse.Namespace) -> None:
    """Inicio rápido: descarga ISOT/LIAR, prepara ISOT y entrena SOLO el modelo productivo."""
    from .data import download_datasets
    from .splits import prepare_dataset
    from .train_v3 import train_final_v3

    download_datasets()
    prepare_dataset()
    out = train_final_v3(quick=True)
    print(json.dumps(out, ensure_ascii=False, default=str))


def _cmd_all(args: argparse.Namespace) -> None:
    for fn in (_cmd_download, _cmd_prepare, _cmd_audit, _cmd_train, _cmd_evaluate, _cmd_analyze):  # solo iteración 1
        fn(args)


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada de la CLI."""
    parser = argparse.ArgumentParser(prog="fakenews", description="Detector de noticias falsas (UCA, IA 2026)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="Descarga ISOT (Kaggle, endpoint público) y LIAR")
    sub.add_parser("prepare", help="Limpieza, deduplicación y splits")
    sub.add_parser("audit", help="Auditoría de calidad y fuga")
    sub.add_parser("train", help="Entrena y selecciona modelos (solo validación)")
    ev = sub.add_parser("evaluate", help="Evaluación final en test (registrada) + LIAR + criterios")
    ev.add_argument("--reason", default="evaluación final", help="Motivo del acceso al test (se registra)")
    sub.add_parser("external", help="Solo validación externa en LIAR")
    sub.add_parser("analyze", help="Clustering y algoritmo genético (exploratorio)")
    pr = sub.add_parser("predict", help="Diagnóstico de un texto (usar '-' para leer de stdin)")
    pr.add_argument("text")
    pr.add_argument("--similar", action="store_true", help="Incluir noticias de train más parecidas")
    sub.add_parser("download-external", help="(it2) Descarga WELFake (Zenodo, CC BY 4.0)")
    sub.add_parser("prepare-external", help="(it2) WELFake: etiquetas verificadas, dedup contra ISOT, dev/holdout")
    sub.add_parser("shortcuts", help="(it2) Auditoría de atajos de un solo rasgo")
    pa = sub.add_parser("panel", help="(it2) Panel de evaluación y selección robusta")
    pa.add_argument("--only", nargs="*", help="Evaluar solo estos candidatos")
    pa.add_argument("--jobs", type=int, default=3)
    tv = sub.add_parser("train-v2", help="(it2) Entrena el modelo final v2 (OOD, abstención)")
    tv.add_argument("--quick", action="store_true", help="Sin externos: reutiliza models/ood_params_v2.json")
    sub.add_parser("evaluate-v2", help="(it2) Evaluación final única: ISOT test + WELFake holdout (registrada)")
    sub.add_parser("format-shortcuts", help="(it3) Descubrimiento automático de atajos de formato")
    p3 = sub.add_parser("panel-v3", help="(it3) Panel con invarianza rigurosa y regla de aceptación")
    p3.add_argument("--only", nargs="*")
    p3.add_argument("--jobs", type=int, default=2)
    t3 = sub.add_parser("train-v3", help="(it3) Modelo final con compuerta de dominio")
    t3.add_argument("--quick", action="store_true", help="Sin externos: reutiliza models/domain_gate_v3.json.gz")
    sub.add_parser("evaluate-v3", help="(it3) Evaluación final de cierre (registrada)")
    sub.add_parser("report", help="(it3) Genera tablas de los documentos desde reports/ y verifica el log")
    sub.add_parser("review-status", help="(cierre) Cuenta los errores revisados y firmados por el equipo")
    sub.add_parser("quickstart", help="Inicio rápido: datos mínimos + solo el modelo productivo")
    al = sub.add_parser("all", help="Todo el pipeline")
    al.add_argument("--reason", default="evaluación final (pipeline completo)")
    args = parser.parse_args(argv)
    _force_utf8_output()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    {"download": _cmd_download, "prepare": _cmd_prepare, "audit": _cmd_audit, "train": _cmd_train,
     "evaluate": _cmd_evaluate, "external": _cmd_external, "analyze": _cmd_analyze,
     "predict": _cmd_predict, "all": _cmd_all, "download-external": _cmd_download_external,
     "prepare-external": _cmd_prepare_external, "shortcuts": _cmd_shortcuts, "panel": _cmd_panel,
     "train-v2": _cmd_train_v2, "evaluate-v2": _cmd_evaluate_v2, "quickstart": _cmd_quickstart,
     "format-shortcuts": _cmd_format_shortcuts, "panel-v3": _cmd_panel_v3, "train-v3": _cmd_train_v3,
     "evaluate-v3": _cmd_evaluate_v3, "report": _cmd_report, "review-status": _cmd_review_status}[args.cmd](args)


if __name__ == "__main__":
    main()
