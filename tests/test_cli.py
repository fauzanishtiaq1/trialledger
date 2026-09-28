import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


def run(*args, data=None):
    return subprocess.run([sys.executable, "-m", "trialledger", *args], cwd=ROOT, input=data, text=True, capture_output=True)


def fixture(tmp_path):
    p = tmp_path / "predictions.jsonl"
    rows = [
        {"id": "a", "y_true": 1, "pred_a": .9, "pred_b": .1, "fold": 0},
        {"id": "b", "y_true": 0, "pred_a": .1, "pred_b": .8, "fold": 1},
        {"id": "c", "y_true": 1, "pred_a": .8, "pred_b": .2, "fold": 0},
        {"id": "d", "y_true": 0, "pred_a": .2, "pred_b": .7, "fold": 1},
    ]
    p.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    return p


def test_help_and_invalid_command_are_clean(tmp_path):
    h = run("--help")
    assert h.returncode == 0 and "compare" in h.stdout and not h.stderr
    bad = run("validate", str(tmp_path / "missing.jsonl"))
    assert bad.returncode == 2 and "internal error" not in bad.stderr.lower()


def test_validate_and_compare_json_output(tmp_path):
    p = fixture(tmp_path)
    v = run("validate", str(p), "--pred", "pred_a", "--pred", "pred_b")
    assert v.returncode == 0, v.stderr
    assert json.loads(v.stdout)["valid"] is True
    c = run("compare", str(p), "--pred", "pred_a", "pred_b", "--bootstrap", "20", "--permutation", "20", "--seed", "7")
    assert c.returncode == 0, c.stderr
    result = json.loads(c.stdout)
    assert result["schema_version"] == "1" and result["n_rows"] == 4
    assert result["bootstrap"]["seed"] == 7


def test_compare_report_path_spaces_and_force(tmp_path):
    p = fixture(tmp_path)
    out = tmp_path / "report with spaces.json"
    args = ("compare", str(p), "--pred", "pred_a", "pred_b", "--bootstrap", "10", "--permutation", "10", "--out", str(out))
    first = run(*args)
    assert first.returncode == 0 and out.exists()
    second = run(*args)
    assert second.returncode == 2
    forced = run(*args, "--force")
    assert forced.returncode == 0


def test_select_never_overwrites_input_even_force(tmp_path):
    p = tmp_path / "scores.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in [
        {"fold": 0, "model": "a", "score": .8}, {"fold": 0, "model": "b", "score": .7},
        {"fold": 1, "model": "a", "score": .6}, {"fold": 1, "model": "b", "score": .9},
    ]) + "\n", encoding="utf-8")
    r = run("select", str(p), "--out", str(p), "--force", "--bootstrap", "20")
    assert r.returncode == 2 and "overwrite" in r.stderr.lower()


class TestCli(unittest.TestCase):
    def test_help_and_invalid(self):
        with __import__('tempfile').TemporaryDirectory() as d:
            test_help_and_invalid_command_are_clean(Path(d))

    def test_validate_compare(self):
        with __import__('tempfile').TemporaryDirectory() as d:
            test_validate_and_compare_json_output(Path(d))

    def test_paths(self):
        with __import__('tempfile').TemporaryDirectory() as d:
            test_compare_report_path_spaces_and_force(Path(d))

    def test_alias(self):
        with __import__('tempfile').TemporaryDirectory() as d:
            test_select_never_overwrites_input_even_force(Path(d))
