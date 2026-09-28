"""Offline, machine-readable CLI for TrialLedger."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import core


class UserError(Exception):
    """An actionable input or argument error."""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trialledger", description="Validate and compare aligned ML predictions.")
    s = p.add_subparsers(dest="command", required=True)
    v = s.add_parser("validate", help="validate a prediction table")
    _common(v, allow_pred=True)
    v.add_argument("--json-out", metavar="PATH")
    v.add_argument("--force", action="store_true")
    v.add_argument("--id-column", default="id")
    c = s.add_parser("compare", help="compare two or more prediction columns")
    _common(c)
    c.add_argument("--pred", nargs="+", required=True, metavar="MODEL")
    c.add_argument("--bootstrap", type=_positive, default=2000)
    c.add_argument("--permutation", type=_positive, default=2000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--alpha", type=_alpha, default=.05)
    c.add_argument("--out", metavar="PATH")
    c.add_argument("--report", choices=("json", "markdown"), default="json")
    c.add_argument("--id-column", default="id")
    c.add_argument("--fold-column", default="fold")
    c.add_argument("--group-column")
    c.add_argument("--force", action="store_true")
    q = s.add_parser("select", help="BBC-F-style selection from fold/model scores")
    q.add_argument("input", metavar="FOLD_SCORES")
    q.add_argument("--model-column", default="model")
    q.add_argument("--fold-column", default="fold")
    q.add_argument("--score-column", default="score")
    q.add_argument("--bootstrap", type=_positive, default=2000)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--alpha", type=_alpha, default=.05)
    q.add_argument("--out", metavar="PATH", default="-")
    q.add_argument("--force", action="store_true")
    return p


def _common(p: argparse.ArgumentParser, *, allow_pred: bool = False) -> None:
    p.add_argument("input", metavar="INPUT", help="CSV/JSONL path, or - for stdin")
    p.add_argument("--format", choices=("auto", "csv", "jsonl"), default="auto")
    p.add_argument("--task", choices=("binary", "regression"), default="binary")
    p.add_argument("--metric", default="roc_auc")
    if allow_pred:
        p.add_argument("--pred", action="extend", nargs="+", dest="preds", metavar="COL")
    p.add_argument("--target", default="y_true")


def _positive(v: str) -> int:
    try: n = int(v)
    except ValueError: raise argparse.ArgumentTypeError("must be an integer")
    if n < 1: raise argparse.ArgumentTypeError("must be positive")
    return n


def _alpha(v: str) -> float:
    try: n = float(v)
    except ValueError: raise argparse.ArgumentTypeError("must be numeric")
    if not 0 < n < 1: raise argparse.ArgumentTypeError("must be between 0 and 1")
    return n


def _load(path: str, fmt: str, preserve_columns=("id", "group", "model")) -> list[dict[str, object]]:
    if path == "-":
        raw = sys.stdin.read()
        if fmt == "csv":
            # Use exactly the file loader's validation and coercion semantics.
            with tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory) / "stdin.csv"
                temporary.write_text(raw, encoding="utf-8")
                return core.load_records(temporary, fmt="csv", preserve_columns=preserve_columns)
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    try: return core.load_records(path, fmt=fmt, preserve_columns=preserve_columns)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as e: raise UserError(str(e)) from e


def _safe_output(path: str | None, text: str, *, force: bool, input_name: str) -> None:
    if path is None or path == "-":
        sys.stdout.write(text)
        return
    out, inp = Path(path), None if input_name == "-" else Path(input_name)
    try:
        if inp and (os.path.samefile(out, inp) if out.exists() else out.resolve() == inp.resolve()):
            raise UserError("output must not overwrite or alias input")
    except FileNotFoundError: pass
    if out.exists() and not force: raise UserError(f"output already exists: {path} (use --force)")
    try:
        if not force:
            with out.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
        else:
            fd, temporary = tempfile.mkstemp(prefix=".trialledger-", dir=out.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                    stream.write(text)
                os.replace(temporary, out)
            finally:
                if os.path.exists(temporary): os.unlink(temporary)
    except OSError as e: raise UserError(f"cannot write output: {e}") from e

def _preflight(path: str | None, *, force: bool, input_name: str) -> None:
    if not path or path == "-": return
    out, inp = Path(path), None if input_name == "-" else Path(input_name)
    try:
        if inp and (os.path.samefile(out, inp) if out.exists() else out.resolve() == inp.resolve()):
            raise UserError("output must not overwrite or alias input")
    except FileNotFoundError: pass
    if out.exists() and not force: raise UserError(f"output already exists: {path} (use --force)")


def _json(value: Any) -> str:
    try: return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    except (TypeError, ValueError) as e: raise UserError(f"result is not valid JSON: {e}") from e


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        a = _parser().parse_args(argv)
        if a.command == "select":
            _preflight(a.out, force=a.force, input_name=a.input)
            rows = _load(a.input, "auto", preserve_columns=(a.model_column, a.fold_column))
            result = core.bootstrap_selection(rows, model_column=a.model_column, fold_column=a.fold_column, score_column=a.score_column, bootstrap=a.bootstrap, seed=a.seed, alpha=a.alpha)
            _safe_output(a.out, _json(result), force=a.force, input_name=a.input)
        else:
            _preflight(a.json_out if a.command == "validate" else a.out, force=a.force, input_name=a.input)
            rows = _load(a.input, a.format, preserve_columns=(a.id_column, "group", getattr(a,"group_column",None)))
            preds = a.preds if a.command == "validate" and a.preds else (a.pred if a.command == "compare" else None)
            validation = core.validate_records(rows, target=a.target, prediction_columns=preds, task=a.task, metric=a.metric, id_column=a.id_column, fold_column=getattr(a,"fold_column","fold"), group_column=getattr(a,"group_column",None))
            if a.command == "validate":
                text = _json(validation)
                _safe_output(a.json_out, text, force=a.force, input_name=a.input)
                return 0 if validation.get("valid") else 2
            if not validation.get("valid"): raise UserError(_json(validation))
            result = core.compare_experiments(rows, prediction_columns=a.pred, target=a.target, task=a.task, metric=a.metric, id_column=a.id_column, fold_column=a.fold_column, group_column=a.group_column, bootstrap=a.bootstrap, seed=a.seed, alpha=a.alpha, permutation=a.permutation)
            rendered = core.render_report(result, format=a.report)
            _safe_output(a.out, rendered if rendered.endswith("\n") else rendered + "\n", force=a.force, input_name=a.input)
        return 0
    except (UserError, ValueError) as e: print(str(e), file=sys.stderr); return 2
    except (OSError, UnicodeError) as e: print(str(e), file=sys.stderr); return 2
    except Exception as e: print(f"internal error: {e}", file=sys.stderr); return 2
