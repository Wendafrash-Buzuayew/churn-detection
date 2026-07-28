# Unified Telecom Churn Engine Design

## Objective

Build `unified_churn_engine.py` to predict `LABEL_CHURN_30D` for telecom subscribers. The engine must surface early and imminent churn while reducing false positives among stable low-frequency paying subscribers and cyclical rechargers.

## Scope and non-negotiable safeguards

- `LABEL_CHURN_30D` is the only supervised target.
- All `LABEL_CHURN_*` columns, identifiers, `DATASET_TYPE`, and known post-outcome fields are excluded from features.
- An OOT-only or unlabeled input is score-only. It is never stratified, pseudo-labeled, or used to train a model.
- Training transformations (numeric fill values, clipping bounds, feature selection, calibration, and threshold selection) are fit on training data only.
- The production decision is precision-first: choose the highest-recall threshold that satisfies a configurable precision floor. If no threshold meets the floor, use the threshold with the highest precision and issue a warning.

## Architecture

The engine has five modular stages:

1. **Schema and split resolution** normalizes column names, validates labels, excludes leakage candidates, and selects explicit TRAIN/TEST, TRAIN/OOT, or a stratified fallback. It uses score-only mode for unlabeled/OOT-only inputs.
2. **Feature engineering** generates vectorized rule meta-features from W1-W13 service activity, revenue, bundles, and tenure. Signals include W13 collapse ratios, terminal zero streaks, recent weighted decay, multi-service collapse breadth, cross-service divergence, recovery, recharge cliffs, stable paid usage, and cyclical recharge patterns.
3. **Model fitting** uses a gradient-boosting classifier with a deterministic sklearn fallback when optional boosting libraries are unavailable. Out-of-fold predictions fit a sigmoid calibration model. This yields calibrated probabilities rather than rank scores.
4. **Decision policy** selects a precision-constrained threshold from calibration predictions. Strong false-positive guards suppress only borderline candidates. High-confidence predictions and terminal multi-service collapse are not suppressed.
5. **Artifacts and reporting** persist a single joblib artifact, output scored rows with model/rule reason codes and tiers, and write metrics and guard-impact reports.

## Feature and decision policy

Rule signals are model inputs, not fixed final decisions. Every input row receives:

- Per-service W13 / W10-W12 baseline ratios and collapse percentages.
- Recent exponentially weighted usage/revenue, W10-W13 slope, and monotone-decay counts.
- Terminal consecutive zero count, W13 zero breadth, and multi-service collapse count.
- Voice/data/bundle divergence to distinguish a genuine broad churn event from a single-service substitution.
- Revenue and bundle recharge-cliff features.
- Stable low-frequency paid-user, recovering-activity, and monthly-cycle guard indicators.
- A transparent rule score and reason flags.

The output class is chosen as follows:

1. Score using calibrated probability.
2. Mark `model_probability >= threshold` as a candidate.
3. Apply a guard only when probability is below a configurable high-confidence threshold and the terminal-collapse override is absent.
4. Assign tiers: imminent, early risk, watchlist, or stable.

## Data handling

Explicit splits are used in this order when labels are present: TRAIN/TEST, TRAIN/OOT, then a stratified random fallback. A test/OOT split missing one class is reported but not used for ranking metrics that require both classes. Training requires at least two target classes and enough minority examples for the chosen cross-validation folds; folds are reduced safely when needed.

All numerical work uses float32 NumPy matrices where possible. Week-matrix operations and zero-run detection are vectorized; row-wise `apply`/`apply_along_axis` is avoided. CSV ingestion supports chunked scoring to limit peak memory.

## Evaluation and acceptance criteria

For labeled evaluation data, report Precision, Recall, F1, ROC-AUC, PR-AUC, flagged volume, and confusion matrix. Also report the number of candidates suppressed by each guard and their observed churn rate when labels exist. The engine must run in score-only mode on an OOT-only file without requiring a label.

## Verification plan

- Compile and import the generated module.
- Run synthetic labeled TRAIN/TEST data through training and evaluation.
- Run an OOT-only unlabeled sample through score-only mode using persisted artifacts.
- Assert no `LABEL_CHURN_*` or `DATASET_TYPE` fields enter the saved feature list.
- Check that terminal multi-service collapse is not suppressed by false-positive guards.
