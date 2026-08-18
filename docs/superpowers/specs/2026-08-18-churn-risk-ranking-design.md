# Churn Risk Ranking Engine Design

**Date:** 2026-08-18
**Status:** Approved for planning

## Why the previous approaches struggled

1. **Wrong framing.** Both `unified_churn_engine.py` and the rules engines treat churn as a
   binary yes/no decision. The unified engine enforces a 0.70 precision floor; at a 4.1%
   churn base rate (Feb data) and ~0.5% (sample data) that threshold is only reachable at
   extreme scores, so it flagged **1 customer out of 2,103** (recall 9%). The rules engine
   went the other way and flagged **103,590 of 521,207 (20%)** with no measured precision.
2. **Trained on the wrong data.** The saved engine was trained on
   `Sample_data_full_feature.csv` — 7,008 rows with only **38 churners**. No model can learn
   a stable decision boundary from 38 positives. Meanwhile 521K labeled rows with 21,207
   churners sit unused in `Feb1_Train_with_recharg.csv`.
3. **Schema drift ignored.** The three CSVs use different names for the same recharge
   features (`RECHARGE_AMT_RECENT_4W` vs `RECHARGE_AMT_TOTAL_4W`, `DAYS_SINCE_LAST_RECHARGE_4W`
   vs `DAYS_SINCE_LAST_RECHARGE`), so features silently vanish depending on the input file.

## Objective

Score every subscriber with a **calibrated 90-day churn probability**, rank them, and cut
the ranking into **capacity-sized action tiers** with *measured* precision/recall/lift on a
held-out snapshot, plus a human-readable reason code per flagged subscriber.

The deliverable is a decision-support ranking, not a binary classifier. The retention team
picks tier sizes from campaign capacity; the engine reports what precision each tier
actually achieves.

## Data reality (verified 2026-08-18)

| File | Rows | Snapshot | Churn rate | Role |
|---|---|---|---|---|
| `Feb1_Train_with_recharg.csv` | 521,207 | 2026-02-01 | 4.07% (21,207) | **Training** |
| `March_validation_with_recharg.csv` | 431,111 | 2026-01-15 | 4.63% (19,960) | **Out-of-sample validation** (~41K MSISDN overlap with train must be excluded from metrics) |
| `Sample_data_full_feature.csv` | 7,008 | 2026-01-15 | 0.54% (38) | Dev fixture / smoke tests only — never for model quality conclusions |

Target: `LABEL_CHURN_90D`. Usage weeks available: W10–W13 plus `*_RECENT_4W` aggregates,
13-week summary stats, and recharge features. AON ≥ 91 in all files.

## Architecture

New package `churn_ranker/` (the old scripts stay untouched as reference):

- `schema.py` — column normalization, alias map to a canonical schema, leakage exclusion
  (`MSISDN*`, `LABEL_*`, `SNAPSHOT_DATE`, `DATASET_TYPE`, date/band strings).
- `features.py` — derived signals computable from the **intersection** schema (W10–W13 +
  recharge): W13/baseline collapse ratios, terminal zero runs, collapse breadth,
  recharge cliff, decay slopes, tenure. Missing inputs become NaN (native NaN support in
  HistGradientBoosting), never fabricated zeros.
- `modeling.py` — `ChurnRanker`: HistGradientBoostingClassifier (already installed,
  sklearn 1.5.2), out-of-fold sigmoid calibration, tier thresholds fixed at train time
  from OOF score quantiles, joblib persistence.
- `evaluation.py` — ranking metrics: ROC-AUC, PR-AUC, and a lift table
  (precision/recall/lift at top 1/2/5/10/20%).
- `tiers.py` — tier assignment from stored thresholds + priority-ordered reason codes.
- `cli.py` — `audit`, `train`, `score` commands; chunked scoring for large files.
- `tests/` — pytest, synthetic fixtures + real-sample smoke test.

## Decision policy

- Tier thresholds are **score quantiles fixed at training time** (stable production volumes):
  Tier 1 = top 1%, Tier 2 = next 4%, Tier 3 = next 10% (configurable).
- Every tiered subscriber gets one reason code, chosen by priority:
  all-services-silent → multi-service collapse → recharge stopped → data collapse →
  voice collapse → gradual decline → model-pattern.
- No suppression guards. Guards existed to patch false positives created by the binary
  framing; with tiers, borderline cases simply land in lower tiers.

## Evaluation & acceptance

- Train on Feb; evaluate on the Jan-15 file with overlapping MSISDNs removed.
- Report ROC-AUC, PR-AUC, lift table, and per-tier precision/recall on validation.
- Acceptance: pipeline runs end-to-end from CLI on the real files; validation lift@top-5%
  materially beats random (baseline expectation ≥ 3× given prior ROC-AUC ~0.71 was achieved
  with 38 positives); all tests pass.
- The audit command documents schema coverage per file, snapshot dates, label rates, and
  MSISDN overlap, so data surprises are recorded, not rediscovered.

## Non-goals

- No re-architecture of raw feature ETL (upstream produces these CSVs).
- No deep-learning / external boosting deps; sklearn only (plus pytest as dev dep).
- No uplift modeling (who to *save* vs who will churn) — future work.
