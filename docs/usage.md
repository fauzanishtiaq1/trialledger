# Usage

## Input schema

Each row needs a unique non-empty ID, finite numeric `y_true`, and at least two finite numeric prediction columns. CSV values are parsed into numbers where possible; JSONL accepts native JSON values. Extra fields are preserved by the loader when requested for diagnostics. Duplicate IDs, missing values, non-finite values, and invalid binary probabilities are errors rather than silently dropped rows.

## Validate and compare

```bash
trialledger validate predictions.csv --format csv --pred baseline --pred candidate --id-column sample_id
trialledger compare predictions.csv --pred baseline candidate --task regression --metric rmse --bootstrap 2000 --permutation 2000 --seed 42 --out results.json
```

`compare` accepts two or more model columns and emits one result for every pair. Supported binary metrics are `accuracy`, `log_loss`, `brier`, and `roc_auc`; regression metrics are `mae`, `rmse`, and `r2`. For loss metrics, the reported score/delta is negated so larger remains better, while `raw_score` preserves the conventional loss.

Output files are never overwritten unless `--force` is supplied. An output path may not alias the input, including with `--force`. Paths containing spaces are supported.

## Fold selection

For a long-form table containing one row per `(fold, model)`:

```bash
trialledger select fold_scores.csv --model-column model --fold-column fold --score-column score --bootstrap 2000 --seed 42 --out selection.json
```

This bootstraps folds, selects the highest in-bag mean, and evaluates that winner on out-of-bag folds. It is a simplified BBC-F-style score-table analysis: it does not refit models, consume row-level prediction matrices, or establish coverage guarantees. Replicates with no out-of-bag fold are reported as invalid and omitted.

## Interpretation checklist

* Inspect effect size and interval together; a p-value is not a license to call a model better.
* Row bootstrap is not robust to clustered/dependent observations or distribution shift.
* Prediction IDs do not prove an untouched out-of-sample provenance chain.
* Pairwise comparisons are not multiplicity-corrected.
* Use an untouched holdout for final claims and retain the JSON report with the input data hash in your experiment record.
