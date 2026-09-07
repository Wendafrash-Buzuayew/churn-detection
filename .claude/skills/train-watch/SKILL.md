---
name: train-watch
description: Launch a churn_ranker training run in the background and watch it to completion, checking process liveness and system memory instead of the user polling by hand with tasklist/wmic/Get-Process. Use when the user wants to kick off training on the full dataset and keep an eye on it.
---

# train-watch

Training on the ~521K-row file takes roughly 20-40 minutes. Rather than
manually re-running `Get-Process`, `tasklist`, and `wmic` in between turns,
this skill launches the run detached, then checks it on an interval via the
`loop` skill's dynamic pacing (`ScheduleWakeup`) until it finishes.

## Launch

Parse `$ARGUMENTS` the same way as the `churn-pipeline` skill (`train_csv`,
`eval_csv`, `artifact`, `report_prefix`; same defaults). Then:

```
nohup ./venv/Scripts/python.exe -m churn_ranker.cli train <train_csv> --eval-csv <eval_csv> --artifact <artifact> --report-prefix <report_prefix> > churn_ranker_outputs/train.log 2>&1 &
echo "PID: $!"
disown
```

Record the PID and the `<report_prefix>_training_report.json` path — these
are what every subsequent check looks at.

## Each check (on wakeup, or when the user asks for status)

1. **Is it done?** Read `<report_prefix>_training_report.json` if it exists —
   the run is complete once it contains a `"validation"` key (the CLI writes
   the base report right after `fit()`, then rewrites it with `"validation"`
   after eval finishes). If done, skip to **Completion** below.

2. **Is the process still alive?**
   ```
   powershell -Command "Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU,StartTime | Format-Table"
   ```
   If the PID from launch is no longer listed and the report file still
   lacks `"validation"`, the run died — check `churn_ranker_outputs/train.log`
   for the error and report it to the user instead of continuing to wait.

3. **Is memory healthy?** (HistGradientBoosting on 521K rows x ~118 features
   is memory-hungry; catching a slow leak early beats discovering an OOM 30
   minutes in.)
   ```
   wmic OS get FreePhysicalMemory,TotalVisibleMemorySize
   ```
   If free memory has dropped below roughly 10% of total, or fell sharply
   since the last check, warn the user — they may want to close other
   applications or reduce `RankerConfig` memory pressure.

4. **Schedule the next check.** Use `ScheduleWakeup` with `noop: true` if
   nothing changed since the last check, `noop: false` if you reported
   something (progress, a warning, completion). Training is long — 300-600s
   between checks is reasonable; don't poll tighter than that.

## Completion

Once `"validation"` appears in the training report:

1. Report the headline metrics to the user (OOF and validation ROC-AUC/PR-AUC).
2. Run `score` and `evaluate` (same commands as steps 3-4 of `churn-pipeline`)
   if the user wants the full pipeline, or stop here if they only asked for
   training.
3. Run the metrics guard from `churn-pipeline` step 5
   (`scripts/compare_training_reports.py compare ...`) against
   `docs/reports/metrics_baseline.json` and surface any regression/drift
   warnings before treating the new artifact as good.
4. Call `ScheduleWakeup` with `stop: true` — the watch is done.
