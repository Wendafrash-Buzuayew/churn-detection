# Telecom Churn Risk Ranking Engine

`churn_ranker` scores every prepaid subscriber with a **calibrated 90-day churn probability**, ranks the base, and cuts the ranking into **capacity-sized action tiers** — each with measured precision/recall and a human-readable reason code the retention team can act on.

**Validated performance** (trained on the Feb-2026 snapshot, evaluated out-of-sample on 390,182 unseen customers):

| Metric | Value |
|---|---|
| ROC-AUC | 0.870 |
| PR-AUC | 0.507 (vs 5.1% base rate) |
| Lift @ top 5% | 9.5× — 48.4% precision, catches 47.3% of all churners |
| Tier 1 precision | 85.4% (6,306 customers, 27% of all churners) |

Full results, per-tier tables, and caveats: [`docs/reports/2026-08-18-churn-ranker-results.md`](docs/reports/2026-08-18-churn-ranker-results.md).

---

## Why ranking instead of binary classification

Two earlier approaches failed for the same underlying reason:

- A **binary ML classifier with a 0.70 precision floor** flagged 1 customer out of 2,103 — at a 4% churn base rate that precision is only reachable at extreme scores, so recall collapses to zero.
- A **rules engine** flagged 103,590 of 521,207 customers (20%) with no measured precision.

Churn intervention is not a yes/no question; it is *"who do we contact first, given campaign capacity?"* This engine answers exactly that: a calibrated risk score, a ranked list, and tiers sized to configurable capacity (default: top 1% / next 4% / next 10%), each with honestly measured precision on held-out data.

The design rationale lives in [`docs/superpowers/specs/2026-08-18-churn-risk-ranking-design.md`](docs/superpowers/specs/2026-08-18-churn-risk-ranking-design.md).

## Quickstart

All commands run from the repo root with the project venv:

```bash
# 1. Profile the input files (rows, churn rates, snapshots, schema coverage, MSISDN overlap)
./venv/Scripts/python.exe -m churn_ranker.cli audit Feb1_Train_with_recharg.csv March_validation_with_recharg.csv --output churn_ranker_outputs/audit_report.json

# 2. Train on one snapshot, evaluate out-of-sample on another
#    (customers appearing in BOTH files are automatically excluded from the evaluation metrics)
./venv/Scripts/python.exe -m churn_ranker.cli train Feb1_Train_with_recharg.csv \
    --eval-csv March_validation_with_recharg.csv \
    --artifact churn_ranker_outputs/churn_ranker.joblib \
    --report-prefix churn_ranker_outputs/churn_ranker

# 3. Score any subscriber file (chunked; works on unlabeled data)
./venv/Scripts/python.exe -m churn_ranker.cli score March_validation_with_recharg.csv \
    --artifact churn_ranker_outputs/churn_ranker.joblib \
    --output churn_ranker_outputs/scores.csv
```

Training on the 521K-row file takes roughly 20–40 minutes (5-fold cross-validation plus a final fit). Everything lands in `churn_ranker_outputs/` (gitignored).

### Score file format

| Column | Meaning |
|---|---|
| `MSISDN`, `MSISDN_9`, `MSISDN_251` | Subscriber identifiers, passed through |
| `churn_probability` | Calibrated probability of churning within 90 days |
| `risk_tier` | `TIER_1_IMMINENT` / `TIER_2_HIGH_RISK` / `TIER_3_WATCHLIST` / `STABLE` |
| `reason_code` | Why the customer is flagged (empty for STABLE) |

Reason codes, in priority order: `all_services_silent_last_week`, `multi_service_collapse`, `recharge_stopped`, `data_usage_collapse`, `voice_usage_collapse`, `gradual_decline`, `model_pattern`.

## How it works

```
input CSV ──► schema.py ──► features.py ──► modeling.py ──► tiers.py ──► scored CSV
             harmonize      derived FE_*     HistGradient-    thresholds
             column names   signals          Boosting + OOF   + reasons
             + drop leakage                  calibration
```

1. **`churn_ranker/schema.py`** — normalizes column names and maps known aliases to a canonical schema (the Feb and March extracts name recharge columns differently, e.g. `RECHARGE_AMT_TOTAL_4W` vs `RECHARGE_AMT_RECENT_4W`). Excludes leakage columns (`MSISDN*`, `LABEL_*`, `SNAPSHOT_DATE`, `DATASET_TYPE`, recharge date/band strings) from the model.
2. **`churn_ranker/features.py`** — derives churn signals from the W10–W13 weekly usage columns per service (data, voice, SMS, bundles, recharge): last-week/baseline collapse ratios, terminal zero-week runs, decay slopes, multi-service collapse breadth, recharge cliffs, tenure. Missing input columns become NaN (never fabricated zeros); HistGradientBoosting handles NaN natively.
3. **`churn_ranker/modeling.py`** — `ChurnRanker`: HistGradientBoostingClassifier, out-of-fold predictions from stratified CV feed a sigmoid calibrator, and tier thresholds are frozen at train time from calibrated OOF score quantiles (so production volumes stay stable). The whole model persists as a single joblib artifact.
4. **`churn_ranker/evaluation.py`** — ranking metrics: ROC-AUC, PR-AUC, and capacity lift tables (precision/recall/lift at top 1/2/5/10/20%).
5. **`churn_ranker/tiers.py`** — tier assignment from stored thresholds and priority-ordered reason codes.
6. **`churn_ranker/cli.py`** — the `audit` / `train` / `score` commands.

Tier sizes are configurable via `RankerConfig.tier_spec` in `modeling.py` — set them from actual campaign capacity.

## Data files

| File | Rows | Snapshot | Churn rate | Role |
|---|---|---|---|---|
| `Feb1_Train_with_recharg.csv` | 521,207 | 2026-02-01 | 4.07% | Training (usage + recharge features) |
| `March_validation_with_recharg.csv` | 431,111 | 2026-01-15 | 4.63% | Out-of-sample validation (~41K MSISDNs overlap with training and are excluded from metrics) |
| `Sample_data_full_feature.csv` | 7,008 | 2026-01-15 | 0.54% (38 churners) | Dev fixture / smoke tests only — never for model-quality conclusions |
| `Feb_Train.csv`, `March_validation.csv` | — | — | — | Older extracts without recharge features; superseded |

Target: `LABEL_CHURN_90D` (1 = churned within 90 days of the snapshot). All CSVs are gitignored — they never enter version control.

> **Known caveat:** the validation snapshot (2026-01-15) *precedes* the training snapshot (2026-02-01), so validation is out-of-sample (disjoint customers) but not strictly out-of-time. When a forward snapshot becomes available, rerun step 2 of the Quickstart against it — no code changes needed.

## Project layout

```
churn_ranker/            the engine (schema, features, modeling, evaluation, tiers, cli)
tests/                   pytest suite (synthetic fixtures + real-sample smoke test)
docs/reports/            training/validation results reports
docs/superpowers/        design spec and implementation plan
churn_ranker_outputs/    generated artifacts: model, reports, scores (gitignored)
venv/                    project virtualenv (pandas 3.0.3, scikit-learn 1.5.2)
```

## Development

```bash
# Run the full test suite (~1-2 min; includes training runs on synthetic data
# and a smoke test on the 7K-row real sample)
./venv/Scripts/python.exe -m pytest tests/ -v
```

- Tests follow TDD; every module has hand-computed expected values in its tests.
- Dependencies are pinned in `requirements.txt`; `pytest` is the only dev-only addition.
- Never commit CSVs, joblib artifacts, or anything under `churn_ranker_outputs/` (enforced by `.gitignore`).

## Operations notes

- **Retraining:** retrain on a fresh labeled snapshot with the Quickstart `train` command; always pass `--eval-csv` with a different snapshot to get honest metrics before promoting the artifact.
- **Tier-volume monitoring:** realized tier volumes run somewhat above nominal (e.g. Tier 1 at 1.47% vs the 1% target) because thresholds come from OOF quantiles while the final model scores sharper — monitor volumes after each retrain.
- **Metric verification:** `score` outputs include *all* subscribers, including any that appeared in training. If you join scores with labels to verify performance, drop training-overlap MSISDNs first (the shipped reports already do).

## Roadmap / known follow-ups

- Logit-space (true Platt) calibration instead of probability-space logistic calibration.
- Warn when label coercion drops unparseable values during training.
- Evaluate on a true forward snapshot when one exists.
- Uplift modeling (who can be *saved*, not just who will churn).
