# TrialLedger examples

Run from this directory's parent project:

```bash
python -m trialledger validate examples/predictions.csv --pred baseline calibrated ensemble
python -m trialledger compare examples/predictions.csv --pred baseline calibrated ensemble --bootstrap 200 --permutation 200 --seed 7 --out /tmp/trialledger-report.json
python -m trialledger select examples/fold_scores.csv --bootstrap 200 --seed 7
```

The CSV keeps IDs as strings (`u001`), includes a held-out fold column, and demonstrates
three-model comparison plus BBC-F-style fold/model selection. The commands are offline,
deterministic for a fixed seed, and write JSON reports suitable for downstream scripts.
