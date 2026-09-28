# TrialLedger contract

TrialLedger is a local, model-agnostic evaluator for comparing candidate ML/AI experiments from out-of-sample predictions. It produces alignment-validated paired metrics, bootstrap uncertainty, and an auditable JSON/Markdown report. It is not a training framework and it must not claim that an observed difference is causal or that it reproduces the cited paper.

## Input contract

The canonical input is UTF-8 JSONL (CSV is accepted by the CLI) with one object per prediction:

```json
{"id":"row-001","y_true":1,"pred_a":0.91,"pred_b":0.62,"fold":0,"group":"site-1"}
```

Required fields are a unique non-empty `id`, finite numeric `y_true`, and at least two finite numeric prediction columns. Optional `fold` is a non-negative integer and optional `group` is a non-empty string. Additional fields are preserved only for validation diagnostics. Missing required values, duplicate IDs, non-finite numbers, mixed target types, or a prediction column with no aligned values are hard errors; rows are never silently dropped. A `--drop-missing` option may be implemented only when it emits the exact dropped IDs and count in the result.

Supported task/metric pairs are `binary` with `accuracy`, `log_loss`, `brier`, or `roc_auc`, and `regression` with `mae`, `rmse`, or `r2`. Binary labels must be 0/1 with both classes present; probabilities must lie in [0,1]. `roc_auc` requires both classes. Metrics are oriented so larger is better in comparisons (losses are negated in the reported delta while their raw value is retained).

## Frozen Python API (`trialledger.core`)

The implementation must expose these signatures and JSON-serialisable return shapes:

```python
def load_records(path: str | pathlib.Path, *, fmt: str = "auto") -> list[dict[str, object]]: ...
def validate_records(records: Sequence[Mapping[str, object]], *, target: str = "y_true", prediction_columns: Sequence[str] | None = None, task: str = "binary", metric: str = "roc_auc", id_column: str = "id") -> dict[str, object]: ...
def compare_experiments(records: Sequence[Mapping[str, object]], *, prediction_columns: Sequence[str], target: str = "y_true", task: str = "binary", metric: str = "roc_auc", id_column: str = "id", fold_column: str | None = "fold", group_column: str | None = None, bootstrap: int = 2000, seed: int = 0, alpha: float = 0.05, permutation: int = 2000) -> dict[str, object]: ...
def render_report(result: Mapping[str, object], *, format: str = "json") -> str: ...
def bootstrap_selection(fold_scores: Sequence[Mapping[str, object]], *, model_column: str = "model", fold_column: str = "fold", score_column: str = "score", bootstrap: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict[str, object]: ...
```

`validate_records` returns `{valid: bool, n_rows: int, ids: {unique: bool, missing: int, duplicates: list[str]}, columns: {...}, errors: list[str], warnings: list[str]}`. `compare_experiments` returns `{schema_version: "1", task, metric, n_rows, models: {name: {score, raw_score, ci_low, ci_high}}, pairwise: [{left, right, delta, delta_ci_low, delta_ci_high, p_permutation, n_pairs}], bootstrap: {method: "paired_row_bootstrap", iterations, seed, alpha}, validation, limitations}`. Every numeric field is finite or `null` with a reason in `limitations`; results include the aligned ID count and any fold/group coverage diagnostics.

The primary uncertainty method is paired row bootstrap: resample aligned IDs with replacement, recompute the metric and model difference, and use percentile bounds. The paired permutation test swaps the two model predictions per ID with probability 1/2 for each permutation and reports a two-sided p-value with +1 correction. If `fold` is provided, a second fold-stratified bootstrap mode may be exposed, but the default must not pretend folds are independent. The paired delta is oriented score(left) minus score(right). Quantiles use linear interpolation at (N-1)*q. Accuracy uses probability >= 0.5; AUC assigns half credit to ties. Group identifiers are diagnostic only: row bootstrap is not cluster bootstrap. No p-value is used as a license to call a model superior without effect size and interval.

`bootstrap_selection` is the paper-inspired BBC-F-style path. Input is long-form fold/model scores with one row per `(fold, model)` and finite scalar `score`; all models must cover the same non-empty folds. Each replicate samples folds with replacement, selects the highest-scoring model using the in-bag fold mean (ties resolve lexicographically), then scores that selected model on out-of-bag folds; the out-of-bag distribution yields the point estimate and percentile interval. This follows the paper's described fold bootstrap/winner-selection structure, but is a simplified score-table implementation: it does not refit models, recreate row-level prediction matrices, or establish coverage guarantees. Replicates with no out-of-bag fold are invalid and omitted; fewer than `max(100, 0.5*bootstrap)` valid replicates is an error. Binary labels are strictly 0/1; log-loss clips probabilities to `[epsilon, 1-epsilon]` with `epsilon=1e-15`; non-finite draws are omitted and counted.

## CLI contract

`python -m trialledger validate INPUT [--format auto|csv|jsonl] [--task ...] [--metric ...] [--pred COL ...] [--json-out PATH]` validates and exits 0 only for valid input; invalid input exits 2.

`python -m trialledger compare INPUT --pred LEFT RIGHT [--task ...] [--metric ...] [--bootstrap N] [--permutation N] [--seed INT] [--alpha FLOAT] [--id-column COL] [--fold-column COL] [--group-column COL] [--out PATH] [--report json|markdown]` writes a deterministic report (except elapsed-time metadata, which is excluded by default) and exits 0 on a valid comparison, 2 on input/argument errors. `python -m trialledger select FOLD_SCORES --model-column model --fold-column fold --score-column score [--bootstrap N] [--seed INT] [--alpha FLOAT] [--out PATH] [--force]` invokes `bootstrap_selection`; it must never overwrite an existing output unless `--force` is supplied, and must NEVER overwrite or alias the input, even with --force. All three commands support --force for explicit replacement of report files. All output paths are checked before writing. Paths with spaces work. No network, paid key, telemetry, or code execution from input is allowed.

## Usefulness benchmark

The validation leaf must compare TrialLedger with a transparent baseline script that computes the same aligned metric and paired permutation on a fixed, versioned synthetic fixture containing known expected scores, duplicate IDs, missing predictions, and an intentionally shuffled second model. It must independently assert exact or tolerance-bounded answers for at least accuracy, RMSE, paired delta, bootstrap reproducibility under a fixed seed, and invalid-input diagnostics. A timing benchmark must measure repeated end-to-end CLI runs on a generated 100,000-row JSONL fixture against the baseline, reporting raw wall-clock medians and environment; it must not be converted into a claimed human time-savings percentage. The report must state that this is a local reproducibility/decision aid, not proof of real-world time saved.

## Rights and attribution

Benchmark bound: the 100,000-row throughput run uses accuracy with `--bootstrap 200 --permutation 200` explicitly (not default 2,000 resamples), and reports raw medians, row count, iterations, seed, and environment. Selection benchmarks separately report fold/model dimensions and valid-draw counts.

Original TrialLedger code is intended for MIT licensing. The cited paper is used as scholarly inspiration and cited by URL; no paper text, figures, code, weights, or datasets are copied. The author-facing arXiv record must be checked for its selected reuse license before redistributing any paper material. A permissive paper license, if present, does not grant patent clearance or rights to third-party dependencies/data. Dependencies retain their own licenses.
