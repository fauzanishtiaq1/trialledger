import csv
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from trialledger.core import bootstrap_selection, compare_experiments, load_records

ROOT = Path(__file__).parents[1]


class TrialLedgerIntegration(unittest.TestCase):
    def _run(self, *args, data=None):
        return subprocess.run([sys.executable, "-m", "trialledger", *args], cwd=ROOT,
                              input=data, text=True, capture_output=True)

    def _csv(self, directory, rows, name="pred.csv"):
        p = Path(directory) / name
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        return p

    def test_numeric_custom_ids_are_not_coerced_or_collided(self):
        data = "sample_id,y_true,a,b\n001,0,0.1,0.2\n1,1,0.9,0.8\n"
        with tempfile.TemporaryDirectory() as td:
            p = self._csv(td, [{"sample_id":"001","y_true":0,"a":.1,"b":.2}, {"sample_id":"1","y_true":1,"a":.9,"b":.8}])
            self.assertEqual(load_records(p, fmt="csv", preserve_columns=("sample_id",))[0]["sample_id"], "001")
            r = self._run("validate", str(p), "--id-column", "sample_id", "--pred", "a", "b")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(json.loads(r.stdout)["ids"]["unique"])

    def test_stdin_csv_and_force_output(self):
        data = "id,y_true,a,b\nx,0,0.1,0.2\ny,1,0.9,0.8\n"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report.json"
            args = ("compare", "-", "--format", "csv", "--pred", "a", "b", "--bootstrap", "20", "--permutation", "20", "--out", str(out))
            first = self._run(*args, data=data); self.assertEqual(first.returncode, 0, first.stderr)
            second = self._run(*args, data=data); self.assertEqual(second.returncode, 2)
            forced = self._run(*args, "--force", data=data); self.assertEqual(forced.returncode, 0, forced.stderr)
            self.assertEqual(json.loads(out.read_text())["n_rows"], 2)

    def test_three_models_and_independent_metric_oracles(self):
        rows = [{"id": str(i), "y_true": i % 2, "a": .9 if i % 2 else .1,
                 "b": .8 if i % 2 else .2, "c": .6 if i % 2 else .4} for i in range(20)]
        result = compare_experiments(rows, prediction_columns=["a", "b", "c"], bootstrap=50, permutation=50, seed=11)
        self.assertEqual(len(result["pairwise"]), 3)
        self.assertAlmostEqual(result["models"]["a"]["score"], 1.0)
        # Independent hand-computed RMSE delta: score is negative RMSE, so a wins.
        reg = [{"id": str(i), "y_true": float(i), "a": float(i), "b": float(i + 1)} for i in range(8)]
        out = compare_experiments(reg, prediction_columns=["a", "b"], task="regression", metric="rmse", bootstrap=40, permutation=40, seed=2)
        self.assertAlmostEqual(out["models"]["a"]["score"], 0.0)
        self.assertAlmostEqual(out["pairwise"][0]["delta"], math.sqrt(1.0), places=12)

    def test_reproducible_permutation_and_tied_auc(self):
        rows = [{"id": str(i), "y_true": i % 2, "a": .5, "b": .5} for i in range(12)]
        x = compare_experiments(rows, prediction_columns=["a", "b"], bootstrap=80, permutation=80, seed=9)
        y = compare_experiments(rows, prediction_columns=["a", "b"], bootstrap=80, permutation=80, seed=9)
        self.assertEqual(x, y); self.assertAlmostEqual(x["models"]["a"]["score"], .5)

    def test_selection_cli_oracle_and_missing_fold_errors(self):
        rows = "fold,model,score\n0,a,0.8\n0,b,0.7\n1,a,0.6\n1,b,0.9\n2,a,0.7\n2,b,0.8\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "folds.csv"; p.write_text(rows, encoding="utf-8")
            r = self._run("select", str(p), "--bootstrap", "200", "--seed", "3")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(r.stdout)["n_folds"], 3)
            bad = Path(td) / "bad.csv"; bad.write_text("fold,model,score\n0,a,0.8\n0,b,0.7\n1,a,0.6\n", encoding="utf-8")
            e = self._run("select", str(bad), "--bootstrap", "10")
            self.assertEqual(e.returncode, 2); self.assertIn("same folds", e.stderr.lower())


if __name__ == "__main__":
    unittest.main()
