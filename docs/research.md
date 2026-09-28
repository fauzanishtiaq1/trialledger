# TrialLedger research basis

## Selected primary source

Konstantinos Paraschakis, Andrea Castellani, Giorgos Borboudakis, and Ioannis Tsamardinos, “Confidence Interval Estimation of Predictive Performance in the Context of AutoML,” arXiv:2406.08099, submitted 12 June 2024. Primary record: https://arxiv.org/abs/2406.08099. The record describes an empirical study over real and simulated datasets and introduces BBC-F (Bootstrap Bias Correction on Folds); it explicitly motivates uncertainty intervals because selecting among many pipelines creates winner’s-curse bias. The paper’s method input is an out-of-sample prediction matrix, not model internals (record lines 18–21 and 23–31 in the HTML rendering).

Actionable mapping: TrialLedger accepts aligned out-of-sample predictions keyed by immutable IDs, computes model metrics, paired row-bootstrap intervals, and paired permutation comparisons. Its separate `bootstrap_selection` path takes fold-by-model scores and performs the paper’s central BBC-F-style operation: bootstrap folds, select the winner from in-bag means, and evaluate that selected winner on out-of-bag folds. This is an independent score-table implementation—not a full BBC-F reproduction—because it does not refit pipelines or consume the paper’s prediction matrix. The report therefore emits limitations and recommends an untouched holdout for final claims.

## Why this product is useful

People routinely compare exported predictions with a spreadsheet or ad-hoc script, lose ID alignment, omit uncertainty, or accidentally compare different rows. A local CLI can make the safe path repeatable: hard validation for duplicate/missing IDs, explicit metric direction, paired effect size, reproducible intervals, and a machine-readable report that can be attached to an experiment. It accepts ordinary CSV/JSONL and needs no account or paid API.

## Independent correctness and usefulness evidence

The downstream validation leaf must create a fixture with independently hand-computed answers and a deliberately adversarial row order. Required checks: shuffled IDs still align; duplicate and missing values fail with diagnostics; perfect and inverted binary predictions produce expected accuracy/AUC; a small regression fixture produces known MAE/RMSE; fixed seed produces byte-stable statistical fields; and CLI JSON agrees with the independent baseline. A 100,000-row generated fixture is used for a timing comparison against the baseline implementation. Timing is evidence about this machine/run only; it is not a user-time-saving claim and must be reported with Python version, OS, row count, iterations, and raw medians.

## Competing tools and boundary

`scikit-learn` supplies metrics and resampling primitives but does not provide this exact auditable input validation/report workflow. `MLflow` and `Weights & Biases` track experiments, but are service-oriented and do not require the same aligned prediction-table contract; W&B may require an account. `Evidently` focuses on data/model monitoring and reports, while `Deepchecks` focuses on validation suites. TrialLedger is intentionally narrower, offline, and dependency-light. It should not market itself as a replacement for those systems or as a statistical certification tool.

## Rights and implementation notes

Competing-tool source URLs: scikit-learn model evaluation https://scikit-learn.org/stable/modules/model_evaluation.html; MLflow tracking https://mlflow.org/docs/latest/ml/tracking/; Weights & Biases docs https://docs.wandb.ai/; Evidently docs https://docs.evidentlyai.com/; Deepchecks docs https://docs.deepchecks.com/.

The primary v1 HTML explicitly lists CC BY 4.0 (https://arxiv.org/html/2406.08099v1, checked 2026-09-28). No paper material is bundled. The arXiv page provides the authoritative citation and version history; the project must record the selected arXiv reuse license if paper artifacts are ever redistributed. arXiv’s general licensing guidance explains that submitters choose a license and that default distribution does not automatically grant broad reuse rights: https://info.arxiv.org/help/license/index.html. Only ideas, equations independently reimplemented, and bibliographic metadata are used here. Original code is MIT; third-party packages and any user-supplied data remain under their own terms. This note is not legal or patent advice and does not guarantee freedom to operate.

## Critic pass / limits

The chosen paper studies confidence intervals after AutoML selection and reports BBC/BBC-F results; TrialLedger’s `bootstrap_selection` reproduces only the fold-score winner-selection skeleton, while paired row bootstrap remains a separate practical comparison mode. Neither mode reproduces the paper’s full pipeline, datasets, or published performance numbers. Bootstrap intervals can be misleading for dependent rows, grouped data, distribution shift, adaptive model selection, or leakage already present in predictions; IDs alone cannot prove predictions were truly out-of-sample. The tool must expose valid/invalid draw counts, preserve raw counts, disclose multiple pairwise comparisons without correction, and avoid “significant = better” wording.
