"""Línea de comandos: `python -m fakenews.cli <comando>`.

Comandos: download, prepare, audit, train, evaluate, external, analyze, predict, all.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import setup_logging

LOGGER = logging.getLogger("fakenews.cli")


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

    text = args.text if args.text != "-" else sys.stdin.read()
    try:
        out = predict(text, with_similar=args.similar)
    except ModelNotTrainedError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def _cmd_all(args: argparse.Namespace) -> None:
    for fn in (_cmd_download, _cmd_prepare, _cmd_audit, _cmd_train, _cmd_evaluate, _cmd_analyze):
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
    al = sub.add_parser("all", help="Todo el pipeline")
    al.add_argument("--reason", default="evaluación final (pipeline completo)")
    args = parser.parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    {"download": _cmd_download, "prepare": _cmd_prepare, "audit": _cmd_audit, "train": _cmd_train,
     "evaluate": _cmd_evaluate, "external": _cmd_external, "analyze": _cmd_analyze,
     "predict": _cmd_predict, "all": _cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
