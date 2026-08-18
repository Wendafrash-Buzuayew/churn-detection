# Churn Ranker Results — trained 2026-08-18

Spec: docs/superpowers/specs/2026-08-18-churn-risk-ranking-design.md

## Setup
- Train: Feb1_Train_with_recharg.csv (521,207 rows, 21,207 churners, snapshot 2026-02-01)
- Validation: March_validation_with_recharg.csv, snapshot 2026-01-15,
  40,929 overlapping MSISDNs removed -> 390,182 rows evaluated
- Model: HistGradientBoostingClassifier + sigmoid calibration, 118 features, 5-fold OOF

## Ranking quality
| Split | ROC-AUC | PR-AUC | Base rate |
|---|---|---|---|
| OOF (train) | 0.858 | 0.394 | 0.041 |
| Validation | 0.870 | 0.507 | 0.051 |

## Validation lift table
| Top % | Contacted | Churners caught | Precision | Recall | Lift |
|---|---|---|---|---|---|
| 1% | 3,902 | 3,740 | 0.958 | 0.187 | 18.737 |
| 2% | 7,804 | 6,144 | 0.787 | 0.308 | 15.390 |
| 5% | 19,509 | 9,446 | 0.484 | 0.473 | 9.465 |
| 10% | 39,018 | 12,077 | 0.310 | 0.605 | 6.051 |
| 20% | 78,036 | 14,774 | 0.189 | 0.740 | 3.701 |

## Reading this
At the validation cohort's 5.1% base rate, random contact catches ~5.1 churners per 100
calls. The lift column
is the multiplier this model achieves at each capacity. Tier sizes (1% / 5% / 15%
cumulative) can be re-fit to actual campaign capacity via RankerConfig.tier_spec.

## Caveats
- The validation snapshot (2026-01-15) precedes the training snapshot (2026-02-01);
  it is out-of-sample (disjoint customers after overlap removal) but not strictly
  out-of-time. A forward snapshot should be evaluated when available.
- If validation lift@5% is below 3.0, treat the model as not yet deployable and
  investigate before promoting (feature drift between files is the first suspect —
  compare `usable_feature_columns` across files in audit_report.json). Here,
  validation lift@5% is 9.465, well above the 3.0 threshold, and validation ROC-AUC
  (0.870) and PR-AUC (0.507) both exceed the OOF training figures, so no drift concern
  is indicated by this run; the caveat about snapshot ordering above still applies.

## Comparison with previous approaches
- unified_churn_engine (binary, precision floor 0.70): flagged 1 of 2,103 eval rows
  (recall 9%), trained on the 38-churner sample.
- churn_rules_v4: flagged 103,590 of 521,207 (19.9%) with no measured precision.
- This ranker: capacity-controlled volumes with measured precision above (e.g. 95.8%
  precision at the top 1% of the validation file, 431,111 rows scored in
  `march_scores.csv` with tier volumes 1.47% / 3.48% / 9.14% for
  TIER_1_IMMINENT / TIER_2_HIGH_RISK / TIER_3_WATCHLIST).
