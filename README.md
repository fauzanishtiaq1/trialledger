# TrialLedger

TrialLedger is a local, model-agnostic evaluator for exported out-of-sample predictions. It validates row alignment, computes comparable metrics, estimates paired uncertainty, and writes JSON or Markdown reports without a service account or runtime dependencies.

## Install

```bash
git clone https://github.com/fauzanishtiaq1/trialledger.git
cd trialledger
python -m pip install .
trialledger --help
```

Python 3.10 or newer is supported. The project intentionally has zero runtime dependencies.

## Three-minute workflow

Given a CSV or JSONL file with `id`, `y_true`, and prediction columns:

```bash
trialledger validate predictions.csv --pred model_a --pred model_b
trialledger compare predictions.csv --pred model_a model_b --metric roc_auc --out report.json
trialledger compare predictions.csv --pred model_a model_b --report markdown --out report.md
```

For more than two candidates, pass all columns after one `--pred` flag. Use `--id-column` when the identifier is not named `id`; `--fold-column` and `--group-column` expose coverage diagnostics. See [docs/usage.md](docs/usage.md) for CSV preservation, regression metrics, and BBC-F-style fold selection.

## What the numbers mean

The default uncertainty is a paired row bootstrap over aligned IDs. Losses are reported with a larger-is-better score orientation while retaining the raw loss. Permutation p-values are descriptive and do not establish superiority; multiple pairwise comparisons are uncorrected. Group labels are diagnostic only, not cluster-robust inference, and IDs cannot prove predictions are genuinely out-of-sample. The fold-selection command is an independent simplified implementation of the winner-selection structure described by Paraschakis et al. (2024), not a reproduction of that paper.

## Citation and license

If you use TrialLedger, cite the metadata in [CITATION.cff](CITATION.cff). Original code is MIT licensed; see [LICENSE](LICENSE). The research basis and source links are documented in `docs/research.md`.

## Included examples and verification

See [examples/README.md](examples/README.md) for shipped prediction and fold-score fixtures. Run `python -m unittest discover -s tests -v` and `python benchmarks/run.py --output benchmarks/results.json`. Raw measured runs and limitations are in [benchmarks/results.json](benchmarks/results.json); no human time savings are claimed.
