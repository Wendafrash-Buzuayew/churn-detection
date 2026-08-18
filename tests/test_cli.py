import json
from pathlib import Path

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
