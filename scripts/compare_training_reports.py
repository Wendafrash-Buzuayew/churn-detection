"""Compare a new retrain's metrics against the last promoted baseline.

Reads the two files the `train`/`evaluate` CLI commands already produce
(`<prefix>_training_report.json` and `evaluate`'s `confusion_matrix.json`) and
either extracts a compact baseline from them or diffs them against a stored
baseline, flagging AUC drops and tier-volume/precision drift the README's
"Operations notes" currently ask a human to check by hand after every retrain.

Usage:
    # After a training + evaluate run you're happy with, save it as the
    # baseline other retrains get compared against:
    python scripts/compare_training_reports.py extract-baseline \\
        --training-report churn_ranker_outputs/churn_ranker_training_report.json \\
        --confusion-matrix churn_ranker_outputs/evaluation/confusion_matrix.json \\
        --train-csv Feb1_Train_with_recharg.csv \\
        --eval-csv March_validation_with_recharg.csv \\
        -o docs/reports/metrics_baseline.json

    # After any later retrain, compare against that baseline:
    python scripts/compare_training_reports.py compare \\
        --baseline docs/reports/metrics_baseline.json \\
        --training-report churn_ranker_outputs/churn_ranker_training_report.json \\
        --confusion-matrix churn_ranker_outputs/evaluation/confusion_matrix.json

Exit code is 0 when nothing crosses a threshold, 1 when something does.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _read_json(path: str | Path) -> dict:
    # utf-8-sig tolerates a BOM (e.g. files written by Windows PowerShell's
    # default UTF-8 encoding), which plain utf-8 would choke on.
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _tier_volumes_pct(confusion_matrix: dict) -> dict[str, float]:
    cohort = confusion_matrix.get("cohort") or 0
    if not cohort:
        return {}
    return {
        row["tier"]: round(100 * row["n"] / cohort, 4)
        for row in confusion_matrix.get("per_tier", [])
    }


def _tier_precision(confusion_matrix: dict, tier: str) -> float | None:
    for row in confusion_matrix.get("per_tier", []):
        if row["tier"] == tier:
            return row.get("precision")
    return None


def extract_metrics(training_report: dict, confusion_matrix: dict | None) -> dict:
    oof = training_report.get("oof_metrics", {})
    validation = training_report.get("validation", {}).get("metrics", {})
    calibration = training_report.get("calibration", {})
    metrics = {
        "n_rows": training_report.get("n_rows"),
        "oof_roc_auc": oof.get("roc_auc"),
        "oof_pr_auc": oof.get("pr_auc"),
        "validation_roc_auc": validation.get("roc_auc"),
        "validation_pr_auc": validation.get("pr_auc"),
        "calibration_source": calibration.get("source"),
        "calibration_brier_score": calibration.get("brier_score"),
        "calibration_ece": calibration.get("ece"),
    }
    if confusion_matrix:
        metrics["tier_volumes_pct"] = _tier_volumes_pct(confusion_matrix)
        metrics["tier1_precision"] = _tier_precision(confusion_matrix, "TIER_1_IMMINENT")
        metrics["overall_precision"] = confusion_matrix.get("precision")
        metrics["overall_recall"] = confusion_matrix.get("recall")
    return metrics


def cmd_extract_baseline(args: argparse.Namespace) -> int:
    training_report = _read_json(args.training_report)
    confusion_matrix = (
        _read_json(args.confusion_matrix) if args.confusion_matrix else None
    )
    baseline = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "train_csv": args.train_csv,
        "eval_csv": args.eval_csv,
        **extract_metrics(training_report, confusion_matrix),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"Wrote baseline to {out_path}")
    print(json.dumps(baseline, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = _read_json(args.baseline)
    training_report = _read_json(args.training_report)
    confusion_matrix = (
        _read_json(args.confusion_matrix) if args.confusion_matrix else None
    )
    current = extract_metrics(training_report, confusion_matrix)

    warnings: list[str] = []

    def check_drop(label: str, base_key: str, threshold: float) -> None:
        base_val = baseline.get(base_key)
        cur_val = current.get(base_key)
        if base_val is None or cur_val is None:
            return
        drop = base_val - cur_val
        if drop > threshold:
            warnings.append(
                f"{label} dropped {drop:.4f} (baseline {base_val:.4f} -> current {cur_val:.4f}, "
                f"threshold {threshold})"
            )

    check_drop("OOF ROC-AUC", "oof_roc_auc", args.auc_drop_threshold)
    check_drop("OOF PR-AUC", "oof_pr_auc", args.pr_auc_drop_threshold)
    check_drop("Validation ROC-AUC", "validation_roc_auc", args.auc_drop_threshold)
    check_drop("Validation PR-AUC", "validation_pr_auc", args.pr_auc_drop_threshold)
    check_drop("Tier 1 precision", "tier1_precision", args.tier1_precision_drop)

    def check_rise(label: str, base_key: str, threshold: float) -> None:
        # Brier score / ECE: lower is better, so a regression is a rise, not a drop.
        base_val = baseline.get(base_key)
        cur_val = current.get(base_key)
        if base_val is None or cur_val is None:
            return
        rise = cur_val - base_val
        if rise > threshold:
            warnings.append(
                f"{label} rose {rise:.4f} (baseline {base_val:.4f} -> current {cur_val:.4f}, "
                f"threshold {threshold})"
            )

    check_rise("Calibration Brier score", "calibration_brier_score", args.brier_score_rise_threshold)
    check_rise("Calibration ECE", "calibration_ece", args.ece_rise_threshold)

    base_volumes = baseline.get("tier_volumes_pct", {})
    cur_volumes = current.get("tier_volumes_pct", {})
    for tier, base_pct in base_volumes.items():
        cur_pct = cur_volumes.get(tier)
        if cur_pct is None:
            continue
        drift = abs(cur_pct - base_pct)
        if drift > args.tier_volume_drift_pp:
            warnings.append(
                f"Tier volume drift for {tier}: {drift:.2f}pp "
                f"(baseline {base_pct:.2f}% -> current {cur_pct:.2f}%, "
                f"threshold {args.tier_volume_drift_pp}pp)"
            )

    print("Baseline:", json.dumps(baseline, indent=2))
    print("Current: ", json.dumps(current, indent=2))
    if warnings:
        print("\nREGRESSION/DRIFT WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
        return 1
    print("\nNo regressions or drift beyond thresholds.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract-baseline", help="Save a training run's metrics as the baseline")
    p_extract.add_argument("--training-report", required=True)
    p_extract.add_argument("--confusion-matrix")
    p_extract.add_argument("--train-csv", required=True)
    p_extract.add_argument("--eval-csv", required=True)
    p_extract.add_argument("-o", "--output", required=True)
    p_extract.set_defaults(func=cmd_extract_baseline)

    p_compare = sub.add_parser("compare", help="Compare a training run's metrics against the baseline")
    p_compare.add_argument("--baseline", required=True)
    p_compare.add_argument("--training-report", required=True)
    p_compare.add_argument("--confusion-matrix")
    p_compare.add_argument("--auc-drop-threshold", type=float, default=0.01)
    p_compare.add_argument("--pr-auc-drop-threshold", type=float, default=0.02)
    p_compare.add_argument("--tier1-precision-drop", type=float, default=0.05)
    p_compare.add_argument("--tier-volume-drift-pp", type=float, default=1.0)
    p_compare.add_argument("--brier-score-rise-threshold", type=float, default=0.005)
    p_compare.add_argument("--ece-rise-threshold", type=float, default=0.02)
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
