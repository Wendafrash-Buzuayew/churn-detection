---
name: churn-pipeline
description: Run the churn_ranker audit -> train -> score -> evaluate pipeline end-to-end per the README Quickstart, then check the new metrics against the committed baseline for regressions/drift. Use when the user wants to retrain, rescore, re-evaluate, or "run the pipeline" for the churn ranker.
---

# churn-pipeline

Runs the four `churn_ranker.cli` stages from the README Quickstart in sequence,
then guards against silent quality regressions using
`scripts/compare_training_reports.py` and the baseline at
`docs/reports/metrics_baseline.json`.

## Arguments

Parse `$ARGUMENTS` as space-separated `key=value` pairs. Defaults come from the
README Quickstart / the current baseline file:

| Key | Default |
|---|---|
| `train_csv` | `Feb1_Train_with_recharg.csv` |
| `eval_csv` | `March_validation_with_recharg.csv` |
| `score_csv` | same as `eval_csv` |
| `artifact` | `churn_ranker_outputs/churn_ranker.joblib` |
| `report_prefix` | `churn_ranker_outputs/churn_ranker` |
| `output_dir` | `churn_ranker_outputs/evaluation` |
| `skip_audit` | `false` |

## Steps

1. **Audit** (unless `skip_audit=true`): confirm `train_csv` and `eval_csv` exist,
   then run
   ```
   ./venv/Scripts/python.exe -m churn_ranker.cli audit <train_csv> <eval_csv> --output churn_ranker_outputs/audit_report.json
   ```
   Skim the JSON for churn rate and schema coverage; flag anything surprising
   (e.g. a churn rate wildly different from the file's historical rate) before
   continuing.

2. **Train.** Training on the full ~521K-row file takes roughly 20-40 minutes.
   Run it in the background (`run_in_background`, or the shell's own
   backgrounding) rather than blocking the turn on it:
   ```
   ./venv/Scripts/python.exe -m churn_ranker.cli train <train_csv> --eval-csv <eval_csv> --artifact <artifact> --report-prefix <report_prefix>
   ```
   Poll for completion (the background task notification, or by checking
   whether `<report_prefix>_training_report.json` contains a `"validation"`
   key) rather than sleep-looping tightly. If you need to actively babysit a
   long run across turns, use the `train-watch` skill instead of polling here.

3. **Score.**
   ```
   ./venv/Scripts/python.exe -m churn_ranker.cli score <score_csv> --artifact <artifact> --output churn_ranker_outputs/scores.csv
   ```

4. **Evaluate.**
   ```
   ./venv/Scripts/python.exe -m churn_ranker.cli evaluate churn_ranker_outputs/scores.csv <score_csv> --train-csv <train_csv> --output-dir <output_dir>
   ```

5. **Metrics guard.** Compare the new run against the committed baseline:
   ```
   ./venv/Scripts/python.exe scripts/compare_training_reports.py compare \
       --baseline docs/reports/metrics_baseline.json \
       --training-report <report_prefix>_training_report.json \
       --confusion-matrix <output_dir>/confusion_matrix.json
   ```
   - Exit code `0`: report the metrics to the user in plain language (ROC-AUC,
     PR-AUC, Tier 1 precision, tier volumes) and note there's no regression.
   - Exit code `1`: **do not silently accept the new artifact.** Show the
     warnings verbatim, and treat the run as not-yet-deployable per the
     README's caveat (lift@5% < 3.0 or any flagged drop) until investigated.

6. **Promote (only if the user confirms).** If metrics are clean (or the user
   explicitly accepts a flagged change) and the user wants this run to become
   the new baseline, run:
   ```
   ./venv/Scripts/python.exe scripts/compare_training_reports.py extract-baseline \
       --training-report <report_prefix>_training_report.json \
       --confusion-matrix <output_dir>/confusion_matrix.json \
       --train-csv <train_csv> --eval-csv <eval_csv> \
       -o docs/reports/metrics_baseline.json
   ```
   This overwrites the committed baseline — treat it like promoting a build,
   not a routine step. Never run it without the user's go-ahead.

## Notes

- All CSVs and `churn_ranker_outputs/` are gitignored; never try to commit them.
- If `train_csv`/`eval_csv` don't exist at the given paths, stop and ask the
  user rather than guessing a substitute file.
