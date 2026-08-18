"""Command-line entry points: audit, train, score."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

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

    args = parser.parse_args(argv)
    if args.command == "audit":
        audit_command(args.csvs, args.output)
    elif args.command == "train":
        train_command(args.train_csv, args.eval_csv, args.artifact, args.report_prefix)
    else:
        score_command(args.input_csv, args.artifact, args.output, args.chunk_size)


if __name__ == "__main__":
    main()
