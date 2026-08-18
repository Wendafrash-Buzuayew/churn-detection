import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn_ranker import cli
from tests.conftest import make_synthetic

SAMPLE = Path(__file__).resolve().parents[1] / "Sample_data_full_feature.csv"


def test_train_then_score_end_to_end(tmp_path, synthetic):
    train_csv = tmp_path / "train.csv"
    synthetic.to_csv(train_csv, index=False)
    artifact = tmp_path / "model.joblib"
    prefix = tmp_path / "report"
    cli.main([
        "train", str(train_csv),
        "--artifact", str(artifact),
        "--report-prefix", str(prefix),
    ])
    assert artifact.exists()
    report = json.loads((tmp_path / "report_training_report.json").read_text())
    assert report["oof_metrics"]["roc_auc"] is not None

    output = tmp_path / "scores.csv"
    cli.main([
        "score", str(train_csv),
        "--artifact", str(artifact),
        "--output", str(output),
        "--chunk-size", "200",
    ])
    scored = pd.read_csv(output)
    assert len(scored) == len(synthetic)
    assert {"MSISDN", "churn_probability", "risk_tier", "reason_code"} <= set(scored.columns)


def test_train_with_eval_removes_msisdn_overlap(tmp_path):
    train = make_synthetic(n=400, seed=1)
    holdout = make_synthetic(n=400, seed=2)
    # give holdout 50 MSISDNs that also exist in train
    holdout.loc[:49, "MSISDN"] = train.loc[:49, "MSISDN"].to_numpy()
    train_csv = tmp_path / "train.csv"
    eval_csv = tmp_path / "eval.csv"
    train.to_csv(train_csv, index=False)
    holdout.to_csv(eval_csv, index=False)
    prefix = tmp_path / "report"
    cli.main([
        "train", str(train_csv),
        "--eval-csv", str(eval_csv),
        "--artifact", str(tmp_path / "model.joblib"),
        "--report-prefix", str(prefix),
    ])
    report = json.loads((tmp_path / "report_training_report.json").read_text())
    assert report["validation"]["overlap_msisdns_removed"] == 50
    assert report["validation"]["metrics"]["n"] == 350
    assert (tmp_path / "report_validation_lift.csv").exists()


def test_audit_reports_files_and_overlap(tmp_path):
    a = make_synthetic(n=100, seed=3)
    b = make_synthetic(n=100, seed=4)
    b.loc[:9, "MSISDN"] = a.loc[:9, "MSISDN"].to_numpy()
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    a.to_csv(path_a, index=False)
    b.to_csv(path_b, index=False)
    output = tmp_path / "audit.json"
    cli.main(["audit", str(path_a), str(path_b), "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["files"]["a.csv"]["rows"] == 100
    assert report["files"]["a.csv"]["churn_rate"] == pytest.approx(0.1)
    assert report["msisdn_overlap"]["a.csv__b.csv"] == 10


def test_train_with_bad_eval_csv_still_writes_report(tmp_path, synthetic):
    train_csv = tmp_path / "train.csv"
    synthetic.to_csv(train_csv, index=False)
    bad_eval = tmp_path / "bad_eval.csv"
    synthetic.drop(columns=["LABEL_CHURN_90D"]).to_csv(bad_eval, index=False)
    prefix = tmp_path / "report"
    with pytest.raises(ValueError):
        cli.main([
            "train", str(train_csv),
            "--eval-csv", str(bad_eval),
            "--artifact", str(tmp_path / "model.joblib"),
            "--report-prefix", str(prefix),
        ])
    report = json.loads((tmp_path / "report_training_report.json").read_text())
    assert report["oof_metrics"]["roc_auc"] is not None
    assert "validation" not in report


def test_score_chunked_matches_unchunked(tmp_path, synthetic):
    train_csv = tmp_path / "train.csv"
    synthetic.to_csv(train_csv, index=False)
    artifact = tmp_path / "model.joblib"
    prefix = tmp_path / "report"
    cli.main([
        "train", str(train_csv),
        "--artifact", str(artifact),
        "--report-prefix", str(prefix),
    ])

    small_chunks = tmp_path / "scores_small.csv"
    big_chunk = tmp_path / "scores_big.csv"
    cli.main([
        "score", str(train_csv),
        "--artifact", str(artifact),
        "--output", str(small_chunks),
        "--chunk-size", "200",
    ])
    cli.main([
        "score", str(train_csv),
        "--artifact", str(artifact),
        "--output", str(big_chunk),
        "--chunk-size", "100000",
    ])

    scored_small = pd.read_csv(small_chunks)
    scored_big = pd.read_csv(big_chunk)
    np.testing.assert_allclose(
        scored_small["churn_probability"], scored_big["churn_probability"]
    )


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample CSV not present")
def test_real_sample_smoke(tmp_path):
    cli.main([
        "train", str(SAMPLE),
        "--artifact", str(tmp_path / "sample.joblib"),
        "--report-prefix", str(tmp_path / "sample"),
    ])
    report = json.loads((tmp_path / "sample_training_report.json").read_text())
    assert report["n_churners"] == 38
    output = tmp_path / "sample_scores.csv"
    cli.main([
        "score", str(SAMPLE),
        "--artifact", str(tmp_path / "sample.joblib"),
        "--output", str(output),
    ])
    assert len(pd.read_csv(output)) == 7008


def test_cli_runs_as_direct_script():
    """python churn_ranker/cli.py must work, not only python -m churn_ranker.cli."""
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "churn_ranker" / "cli.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(repo_root / "churn_ranker"),
    )
    assert result.returncode == 0, result.stderr
    assert "audit" in result.stdout


def test_evaluate_writes_confusion_matrix_and_error_lists(tmp_path, synthetic):
    train_csv = tmp_path / "train.csv"
    synthetic.to_csv(train_csv, index=False)
    artifact = tmp_path / "model.joblib"
    cli.main([
        "train", str(train_csv),
        "--artifact", str(artifact),
        "--report-prefix", str(tmp_path / "report"),
    ])
    scores = tmp_path / "scores.csv"
    cli.main(["score", str(train_csv), "--artifact", str(artifact), "--output", str(scores)])

    out_dir = tmp_path / "evaluation"
    cli.main(["evaluate", str(scores), str(train_csv), "--output-dir", str(out_dir)])

    report = json.loads((out_dir / "confusion_matrix.json").read_text())
    cm = report["confusion_matrix"]
    assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == len(synthetic)
    assert cm["tp"] + cm["fn"] == int(synthetic["LABEL_CHURN_90D"].sum())
    assert report["action_tiers"] == ["TIER_1_IMMINENT", "TIER_2_HIGH_RISK"]

    assert report["actual_churners"] == cm["tp"] + cm["fn"]
    assert report["caught_churners"] == cm["tp"]
    assert report["missed_churners"] == cm["fn"]
    assert sum(report["missed_by_tier"].values()) == cm["fn"]
    assert all(tier not in report["action_tiers"] for tier in report["missed_by_tier"])

    tier_table = pd.read_csv(out_dir / "per_tier_table.csv")
    assert list(tier_table.columns) == [
        "tier", "contacted", "customers", "actual_churners",
        "precision_pct", "share_of_all_churners_pct", "cumulative_recall_pct",
    ]
    assert len(tier_table) == 4
    assert tier_table["actual_churners"].sum() == cm["tp"] + cm["fn"]
    assert tier_table["cumulative_recall_pct"].iloc[-1] == pytest.approx(100.0)
    markdown = (out_dir / "per_tier_table.md").read_text()
    assert "TIER_1_IMMINENT" in markdown and "|" in markdown

    missed = pd.read_csv(out_dir / "missed_churners.csv")
    false_positives = pd.read_csv(out_dir / "false_positives.csv")
    assert len(missed) == cm["fn"]
    assert len(false_positives) == cm["fp"]
    expected_columns = ["MSISDN", "churn_probability", "risk_tier", "reason_code"]
    assert list(missed.columns) == expected_columns
    assert list(false_positives.columns) == expected_columns
    # error lists are sorted most-likely-churner first
    assert missed["churn_probability"].is_monotonic_decreasing
