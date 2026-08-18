"""Command-line entry points: audit, train, score."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

try:
    from churn_ranker import evaluation, schema
    from churn_ranker.modeling import ChurnRanker
except ImportError:
    # Executed as a direct script (python churn_ranker/cli.py): Python puts the
    # package directory, not the repo root, on sys.path — add the root and retry.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from churn_ranker import evaluation, schema
    from churn_ranker.modeling import ChurnRanker


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def audit_command(paths: list[str], output: str) -> dict:
    files: dict[str, dict] = {}
    msisdns: dict[str, set] = {}
    for path in paths:
        df = schema.to_canonical(pd.read_csv(path))
        name = Path(path).name
        target_present = schema.TARGET in df.columns
        numeric = set(df.select_dtypes(include="number").columns)
        files[name] = {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "snapshot_dates": (
                df["SNAPSHOT_DATE"].astype(str).value_counts(dropna=False).to_dict()
                if "SNAPSHOT_DATE" in df.columns else {}
            ),
            "churn_rate": (
                float(pd.to_numeric(df[schema.TARGET], errors="coerce").mean())
                if target_present else None
            ),
            "usable_feature_columns": len(schema.feature_columns(df)),
            "non_numeric_columns": [c for c in df.columns if c not in numeric],
        }
        if "MSISDN" in df.columns:
            msisdns[name] = set(df["MSISDN"].astype(str))
        del df
    overlap = {
        f"{a}__{b}": len(msisdns[a] & msisdns[b])
        for a, b in combinations(sorted(msisdns), 2)
    }
    report = {"files": files, "msisdn_overlap": overlap}
    _ensure_parent(output)
    Path(output).write_text(json.dumps(report, indent=2, default=str))
    return report


def train_command(train_csv: str, eval_csv: str | None,
                  artifact: str, report_prefix: str) -> dict:
    train_df = pd.read_csv(train_csv)
    ranker = ChurnRanker()
    summary = ranker.fit(train_df)
    _ensure_parent(artifact)
    ranker.save(artifact)
    # Write base training report immediately after fit/save to preserve it even if eval fails
    report_path = f"{report_prefix}_training_report.json"
    _ensure_parent(report_path)
    Path(report_path).write_text(json.dumps(summary, indent=2))
    if eval_csv:
        eval_df = pd.read_csv(eval_csv)
        train_norm = schema.normalize_columns(train_df)
        eval_norm = schema.normalize_columns(eval_df)
        overlap_removed = 0
        if "MSISDN" in train_norm.columns and "MSISDN" in eval_norm.columns:
            seen = set(train_norm["MSISDN"].astype(str))
            keep = ~eval_norm["MSISDN"].astype(str).isin(seen)
            overlap_removed = int((~keep).sum())
            eval_df = eval_df.loc[keep.to_numpy()].reset_index(drop=True)
        eval_canonical = schema.to_canonical(eval_df)
        if schema.TARGET not in eval_canonical.columns:
            raise ValueError(f"--eval-csv must contain {schema.TARGET}")
        y_eval = (
            pd.to_numeric(eval_canonical[schema.TARGET], errors="coerce")
            .fillna(0).astype(int).to_numpy()
        )
        scores = ranker.predict_proba(eval_df)
        lift = evaluation.lift_table(y_eval, scores)
        _ensure_parent(f"{report_prefix}_validation_lift.csv")
        lift.to_csv(f"{report_prefix}_validation_lift.csv", index=False)
        summary["validation"] = {
            "overlap_msisdns_removed": overlap_removed,
            "metrics": evaluation.ranking_metrics(y_eval, scores),
            "lift_table": lift.to_dict(orient="records"),
        }
        # Rewrite report with validation block after successful eval
        Path(report_path).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary.get("oof_metrics", {}), indent=2))
    return summary


def score_command(input_csv: str, artifact: str, output: str, chunk_size: int) -> None:
    ranker = ChurnRanker.load(artifact)
    _ensure_parent(output)
    first = True
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
        scored = ranker.score(chunk)
        id_columns = [c for c in scored.columns if str(c).upper().startswith("MSISDN")]
        keep = id_columns + ["churn_probability", "risk_tier", "reason_code"]
        scored[keep].to_csv(output, mode="w" if first else "a", header=first, index=False)
        first = False
    print(f"Wrote {output}")


DEFAULT_ACTION_TIERS = ("TIER_1_IMMINENT", "TIER_2_HIGH_RISK")
ERROR_LIST_COLUMNS = ["MSISDN", "churn_probability", "risk_tier", "reason_code"]


def evaluate_command(scores_csv: str, labels_csv: str, train_csv: str | None,
                     action_tiers: list[str], output_dir: str) -> dict:
    """Confusion matrix plus named error lists for a scored file against labels.

    "Predicted churner" means the customer landed in one of the action tiers —
    the customers a retention campaign would actually contact.
    """
    scored = schema.normalize_columns(pd.read_csv(scores_csv))
    labeled = schema.normalize_columns(pd.read_csv(labels_csv))
    missing = {"MSISDN", "CHURN_PROBABILITY", "RISK_TIER", "REASON_CODE"} - set(scored.columns)
    if missing:
        raise ValueError(f"scores file missing columns: {sorted(missing)}")
    if "MSISDN" not in labeled.columns or schema.TARGET not in labeled.columns:
        raise ValueError(f"labels file must contain MSISDN and {schema.TARGET}")

    scored["MSISDN"] = scored["MSISDN"].astype(str)
    labels = labeled[["MSISDN", schema.TARGET]].copy()
    labels["MSISDN"] = labels["MSISDN"].astype(str)
    merged = scored.merge(labels, on="MSISDN", how="inner")

    overlap_removed = 0
    if train_csv:
        train_ids = set(schema.normalize_columns(pd.read_csv(train_csv))["MSISDN"].astype(str))
        keep = ~merged["MSISDN"].isin(train_ids)
        overlap_removed = int((~keep).sum())
        merged = merged.loc[keep]

    y = pd.to_numeric(merged[schema.TARGET], errors="coerce").fillna(0).astype(int).to_numpy()
    contacted = merged["RISK_TIER"].isin(action_tiers).to_numpy()
    tp = int((contacted & (y == 1)).sum())
    fp = int((contacted & (y == 0)).sum())
    fn = int((~contacted & (y == 1)).sum())
    tn = int((~contacted & (y == 0)).sum())

    tier_order = [name for name, _ in ChurnRanker().config.tier_spec] + ["STABLE"]
    per_tier = []
    total_churners = max(int(y.sum()), 1)
    for tier in tier_order:
        in_tier = (merged["RISK_TIER"] == tier).to_numpy()
        n = int(in_tier.sum())
        churners = int((in_tier & (y == 1)).sum())
        per_tier.append({
            "tier": tier,
            "n": n,
            "churners": churners,
            "precision": churners / n if n else None,
            "share_of_all_churners": churners / total_churners,
        })

    missed_by_tier = {
        entry["tier"]: entry["churners"]
        for entry in per_tier
        if entry["tier"] not in action_tiers
    }
    report = {
        "scores_csv": scores_csv,
        "labels_csv": labels_csv,
        "cohort": int(len(merged)),
        "actual_churners": int(y.sum()),
        "caught_churners": tp,
        "missed_churners": fn,
        "missed_by_tier": missed_by_tier,
        "overlap_msisdns_removed": overlap_removed,
        "action_tiers": list(action_tiers),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "per_tier": per_tier,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "confusion_matrix.json").write_text(json.dumps(report, indent=2))
    error_frame = merged.rename(columns={
        "CHURN_PROBABILITY": "churn_probability",
        "RISK_TIER": "risk_tier",
        "REASON_CODE": "reason_code",
    })[ERROR_LIST_COLUMNS + [schema.TARGET]]
    churner = pd.to_numeric(error_frame[schema.TARGET], errors="coerce").fillna(0).astype(int) == 1
    in_action = error_frame["risk_tier"].isin(action_tiers)
    for name, mask in (("missed_churners.csv", churner & ~in_action),
                       ("false_positives.csv", ~churner & in_action)):
        error_frame.loc[mask, ERROR_LIST_COLUMNS].sort_values(
            "churn_probability", ascending=False
        ).to_csv(output / name, index=False)

    total_caught = 0
    table_rows = []
    for entry in per_tier:
        total_caught += entry["churners"]
        table_rows.append({
            "tier": entry["tier"],
            "contacted": "yes" if entry["tier"] in action_tiers else "no",
            "customers": entry["n"],
            "actual_churners": entry["churners"],
            "precision_pct": round(100 * entry["precision"], 1) if entry["precision"] is not None else 0.0,
            "share_of_all_churners_pct": round(100 * entry["share_of_all_churners"], 1),
            "cumulative_recall_pct": round(100 * total_caught / total_churners, 1),
        })
    tier_table = pd.DataFrame(table_rows)
    tier_table.to_csv(output / "per_tier_table.csv", index=False)
    header = "| " + " | ".join(tier_table.columns) + " |"
    separator = "|" + "|".join(["---"] * len(tier_table.columns)) + "|"
    markdown_rows = ["| " + " | ".join(str(value) for value in row) + " |"
                     for row in tier_table.itertuples(index=False)]
    (output / "per_tier_table.md").write_text("\n".join([header, separator, *markdown_rows]) + "\n")

    contacted_total = tp + fp
    actual = report["actual_churners"]
    print(f"Cohort: {report['cohort']:,} customers"
          + (f" ({overlap_removed:,} training-overlap customers excluded)" if overlap_removed else ""))
    print(f"Actual churners in cohort: {actual:,}")
    if actual:
        print(f"  Caught by action tiers: {tp:,} ({tp / actual:.1%})")
        print(f"  Missed: {fn:,} ({fn / actual:.1%})")
        for tier, count in missed_by_tier.items():
            note = " (scored stable - truly missed)" if tier == "STABLE" else " (still visible, below the action line)"
            print(f"    - in {tier}: {count:,}{note}")
    print(f"Action tiers {list(action_tiers)} -> contact {contacted_total:,} customers")
    print(f"  Caught churners (TP): {tp:,}   Wasted contacts (FP): {fp:,}")
    print(f"  Missed churners (FN): {fn:,}   Correctly left alone (TN): {tn:,}")
    if contacted_total:
        print(f"  Precision {report['precision']:.1%} | Recall {report['recall']:.1%}")
    print()
    print(tier_table.to_string(index=False, formatters={
        "customers": "{:,}".format, "actual_churners": "{:,}".format,
    }))
    print()
    print(f"Wrote {output / 'confusion_matrix.json'}, per_tier_table.csv/.md, "
          "missed_churners.csv, false_positives.csv")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="churn_ranker")
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="Profile CSVs and cross-file MSISDN overlap")
    p_audit.add_argument("csvs", nargs="+")
    p_audit.add_argument("--output", default="churn_ranker_outputs/audit_report.json")

    p_train = sub.add_parser("train", help="Fit ranker; optionally evaluate out-of-sample")
    p_train.add_argument("train_csv")
    p_train.add_argument("--eval-csv", default=None)
    p_train.add_argument("--artifact", default="churn_ranker_outputs/churn_ranker.joblib")
    p_train.add_argument("--report-prefix", default="churn_ranker_outputs/churn_ranker")

    p_score = sub.add_parser("score", help="Score a CSV in chunks")
    p_score.add_argument("input_csv")
    p_score.add_argument("--artifact", default="churn_ranker_outputs/churn_ranker.joblib")
    p_score.add_argument("--output", default="churn_ranker_outputs/churn_scores.csv")
    p_score.add_argument("--chunk-size", type=int, default=100_000)

    p_eval = sub.add_parser("evaluate", help="Confusion matrix and error lists for a scored file vs labels")
    p_eval.add_argument("scores_csv")
    p_eval.add_argument("labels_csv")
    p_eval.add_argument("--train-csv", default=None,
                        help="Exclude customers present in this training file from the metrics")
    p_eval.add_argument("--action-tiers", default=",".join(DEFAULT_ACTION_TIERS),
                        help="Comma-separated tiers counted as 'predicted churner'")
    p_eval.add_argument("--output-dir", default="churn_ranker_outputs/evaluation")

    args = parser.parse_args(argv)
    if args.command == "audit":
        audit_command(args.csvs, args.output)
    elif args.command == "train":
        train_command(args.train_csv, args.eval_csv, args.artifact, args.report_prefix)
    elif args.command == "evaluate":
        evaluate_command(args.scores_csv, args.labels_csv, args.train_csv,
                         [t.strip() for t in args.action_tiers.split(",") if t.strip()],
                         args.output_dir)
    else:
        score_command(args.input_csv, args.artifact, args.output, args.chunk_size)


if __name__ == "__main__":
    main()
