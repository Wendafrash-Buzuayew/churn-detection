# Churn Ranker — Improvement Roadmap

**Date:** 2026-08-19
**Status of the system today:** trained on the 2026-02-01 snapshot (521,207 subscribers),
validated on 390,182 unseen customers — ROC-AUC 0.870, lift@5% 9.5×, Tier-1 precision 85.4%.

This document is the plan for making the system *provably better every quarter*, ordered by
payoff per effort.

## 1. Prove it forward in time — next 1–3 months, near-zero effort

Current validation is honest but backwards: the validation snapshot (2026-01-15) *predates*
the training snapshot (2026-02-01). The single most credibility-building step is a **true
out-of-time test**: take a snapshot from ~May 2026 (its 90-day label window has now matured)
and run, with zero code changes:

```powershell
python -m churn_ranker.cli train Feb1_Train_with_recharg.csv --eval-csv <may_snapshot.csv>
```

If lift@5% holds near 9×, the model is deployment-proven. If it drops, we learned about
drift before management did.

## 2. Retrain rhythm with a metrics ledger — ongoing

Churn behavior drifts with pricing, promos, and seasons. Retrain monthly or quarterly:

1. `audit` the new extract — it already flags schema drift and churn-rate shifts.
2. Retrain and evaluate against the previous snapshot.
3. **Append each run's ROC-AUC / PR-AUC / lift@5% / per-tier precision to a running log**
   (one row per retrain, e.g. `docs/reports/retrain-ledger.csv`).
4. Promote the new artifact only when it beats or matches the champion on the same
   evaluation cohort (champion/challenger).

The ledger doubles as the management story over time: "9.5× in August, X× in November."

## 3. Close the campaign feedback loop — start with the FIRST campaign

Once the retention team contacts Tier 1/2 customers, two things happen:

- **We can finally measure money**: save rate and revenue retained per tier — worth more
  in a management deck than any ML metric.
- **Labels get contaminated**: a customer we contacted and *saved* looks like a false
  positive in the next training set, so the model is punished for its successes.
  **Log every intervention** (MSISDN, date, offer, outcome) and either exclude treated
  customers from future training or add a `was_contacted` feature. Skipping this quietly
  degrades every future retrain.

## 4. Richer signals — the biggest accuracy headroom

The model currently sees only 4 raw weeks (W10–W13) plus aggregates. Candidate additions,
each cheap to test because the pipeline picks up new numeric columns automatically:

- Full 13-week weekly series instead of W10–W13 only
- Complaint / call-center contact counts
- Network experience (call drop rates, data session failures)
- Balance trajectory
- Device change events
- Port-out (MNP) requests, if available

Add one signal group per retrain; keep it only if validation lift improves. The `evaluate`
command produces the before/after per-tier table for free.

## 5. Model-side tuning — after the data improvements

- Logit-space (true Platt) calibration instead of probability-space logistic calibration
- LightGBM / XGBoost comparison against HistGradientBoosting
- Hyperparameter search
- Warn when label coercion drops unparseable values during training

Expect single-digit-percent gains — data improvements (#4) usually beat all of these combined.

## 6. End-game: uplift modeling

Today the model answers *"who will churn?"* The more valuable question is *"who will churn
**but can be saved by a call**?"* Some churners leave no matter what; some stayers never
needed the call; contacting either wastes budget.

Uplift modeling requires a **random control group in every campaign** (e.g. 10% of Tier 1
deliberately not contacted). Start collecting that control data with the very first
campaign — it costs almost nothing now and is impossible to reconstruct later.

## Monitoring checklist per scoring run

- `audit` the extract: row count, churn rate (if labeled), schema coverage vs training
- Tier volumes vs nominal (1% / 4% / 10%) — drift beyond ~2× nominal means recalibrate
- Score distribution shift vs previous month (quantile comparison)

## If we do only three things

1. Run the forward-snapshot validation as soon as May-2026 labels mature.
2. Start the intervention log + random control group with the first campaign.
3. Keep the retrain metrics ledger.

Those three turn a good one-off model into a system that provably improves every quarter.
