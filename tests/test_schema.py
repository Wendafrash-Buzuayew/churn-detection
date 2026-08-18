import pandas as pd

from churn_ranker import schema


def test_normalize_columns_uppercases_and_strips():
    df = pd.DataFrame({" aon ": [1], "data_mb_w13": [2.0]})
    out = schema.normalize_columns(df)
    assert list(out.columns) == ["AON", "DATA_MB_W13"]


def test_alias_renamed_to_canonical():
    df = pd.DataFrame({"recharge_amt_total_4w": [1.0], "AON": [100]})
    out = schema.to_canonical(df)
    assert "RECHARGE_AMT_RECENT_4W" in out.columns
    assert "RECHARGE_AMT_TOTAL_4W" not in out.columns


def test_canonical_name_wins_over_alias():
    df = pd.DataFrame({
        "RECHARGE_AMT_RECENT_4W": [2.0],
        "RECHARGE_AMT_TOTAL_4W": [1.0],
    })
    out = schema.to_canonical(df)
    assert out["RECHARGE_AMT_RECENT_4W"].tolist() == [2.0]
    assert "RECHARGE_AMT_TOTAL_4W" in out.columns  # untouched, just not renamed


def test_feature_columns_exclude_leakage_and_non_numeric():
    df = pd.DataFrame({
        "MSISDN": [251900000001],
        "MSISDN_9": [900000001],
        "LABEL_CHURN_90D": [0],
        "SNAPSHOT_DATE": ["2026-01-15"],
        "DATASET_TYPE": ["TRAIN"],
        "LAST_RECHARGE_RECENCY_BAND": ["0-7D"],
        "DATA_MB_W13": [5.0],
        "AON": [120],
    })
    cols = schema.feature_columns(schema.to_canonical(df))
    assert sorted(cols) == ["AON", "DATA_MB_W13"]
