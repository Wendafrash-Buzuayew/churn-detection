# Blind Test — Scoring Real Data With No Labels or Flags

**Date:** 2026-08-19
**Question:** if we hand the engine real subscriber data with the churn label and all
train/test flags stripped out, does it still identify the actual churners?

## Method

1. Took the real 431,111-row validation extract and **removed `LABEL_CHURN_90D` and
   `DATASET_TYPE` entirely** (verified: no label or flag columns remained — 100 columns vs 102).
2. Scored the blind file with the production artifact (trained on the 2026-02-01 snapshot):

   ```powershell
   python -m churn_ranker.cli score churn_ranker_outputs\blind_test.csv --artifact churn_ranker_outputs\churn_ranker.joblib --output churn_ranker_outputs\blind_scores.csv
   ```

3. Only then revealed the held-back labels and measured the blind predictions against them,
   excluding the 40,929 customers the model had seen in training:

   ```powershell
   python -m churn_ranker.cli evaluate churn_ranker_outputs\blind_scores.csv March_validation_with_recharg.csv --train-csv Feb1_Train_with_recharg.csv --output-dir churn_ranker_outputs\blind_evaluation
   ```

4. Cross-checked that the blind scores are **bit-identical** to scores produced from the
   labeled file (431,111 of 431,111 rows: same probabilities, same tiers) — the label and
   flag columns have zero influence on scoring.

## Result: yes — it identifies them

Blind cohort after overlap exclusion: **390,182 customers the model had never seen,
19,960 of whom actually churned** (unknown to the engine at scoring time).

| tier | contacted | customers | actual churners | precision % | share of all churners % | cumulative recall % |
|---|---|---|---|---|---|---|
| TIER_1_IMMINENT | yes | 6,306 | 5,384 | 85.4 | 27.0 | 27.0 |
| TIER_2_HIGH_RISK | yes | 14,486 | 4,330 | 29.9 | 21.7 | 48.7 |
| TIER_3_WATCHLIST | no | 36,969 | 3,863 | 10.4 | 19.4 | 68.0 |
| STABLE | no | 332,421 | 6,383 | 1.9 | 32.0 | 100.0 |

- Scoring blind, the engine put **5,384 real churners in its top tier of 6,306 (85.4% precision)** —
  against a 5.1% base rate, that is ~17× better than random.
- The two action tiers (5.3% of the base) contained **48.7% of all actual churners**.
- Of the churners it "missed", 3,863 were still visible on the watchlist; only 6,383 of
  19,960 (32%) were scored stable.
- Random contact at the same volume (20,792 customers) would have caught ~1,065 churners;
  the blind engine caught **9,714**.

## What this proves

- **No leakage:** predictions do not depend on the label, the dataset flag, or any
  train/test bookkeeping — those columns can be absent entirely.
- **Production-shape workflow:** this is exactly how live scoring will run — a raw,
  unlabeled subscriber extract in, ranked tiers with reason codes out.
- **Repeatable:** rerun the three commands above against any future unlabeled extract;
  once its labels mature (90 days), run step 3 to grade the predictions.

Artifacts (gitignored, regenerable): `churn_ranker_outputs\blind_test.csv`,
`blind_scores.csv`, `blind_evaluation\` (confusion matrix, per-tier table,
missed-churner and false-positive lists).
