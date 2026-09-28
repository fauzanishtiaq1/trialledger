# TrialLedger adversarial review (2026-09-28)

## Scope and evidence

This review was run from a clean Python 3.12.10 virtual environment created outside
the repository. `python -m pip install .` completed, the installed `trialledger --help`
entry point ran, and the installed package passed the repository's standard-library
suite: **16 tests, 16 passed, 0 failed** (`python -m unittest discover -s tests -v`).
The shipped example commands in `examples/README.md` were inspected and the full
benchmark was run with its defaults (100,000 rows, 200 bootstrap/permutation draws,
3 repetitions). It printed `BENCHMARK_OK` and independently matched the accuracy
baseline and fold-selection oracle on every repetition.

Measured benchmark on Windows 11, AMD64, Python 3.12.10:

* tool median: 9.0817 s; specialized baseline median: 4.3351 s;
* all three accuracy point estimates, CIs, deltas, and permutation p-values matched
  the independent baseline to 1e-12;
* fold selection: 483 valid and 17 invalid draws, point estimate
  0.7232229123533471, matching the independent oracle to 1e-12.

This is a correctness/runtime benchmark only. It does not establish human-hours saved
or a general speedup. The benchmark output explicitly records those limitations.

## Re-review after fixes (2026-09-28)

The prior findings are resolved. A targeted regression test now passes for custom fold
validation and coverage. A positive CLI run using `--fold-column cv --group-column grp`
returned diagnostics with `column`, `missing`, `unique`, and per-value `counts` for
both fields. A negative run with `cv=-1` exited with validation error `cv must be a
non-negative integer`. No new defect was reproduced.

## Checks of requested failure surfaces

The existing tests and source review covered three-model pair counts, tied ROC-AUC,
loss orientation with raw loss retained, constant-target R2 rejection, finite-value
validation, JSON serialization with `allow_nan=False`, custom/numeric IDs, CSV001
string ID preservation, output overwrite refusal, and input alias protection. No
additional failure was reproduced in those areas. The selection implementation checks
complete rectangular fold/model coverage and reports invalid no-OOB draws.

## Research/claims review

`docs/research.md` identifies Paraschakis et al. (2024) and describes the implementation
as a simplified, independent BBC-F-style fold-score skeleton. The code and README do
not claim reproduction or coverage guarantees, and the report limitations state that
there is no refitting or external-performance guarantee. Keep that language: this
review found no basis for a stronger claim. The repository is MIT-licensed and the
research link is separately cited; this review did not find copied paper text/code.

## Release disposition

The core calculations, diagnostics, custom-column validation, fresh-install workflow,
and documented claims are reproducible. This re-review is **release-clean** subject to
the parent branch's full benchmark rerun; no unresolved release-blocking finding
remains in this leaf.
