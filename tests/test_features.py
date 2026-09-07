import numpy as np
import pandas as pd
import pytest

from churn_ranker import features

PREFIXES = ("DATA_MB", "OG_VOICE_MIN", "TOTAL_SMS_COUNT", "BUNDLE_CNT", "RECHARGE_AMT")


def frame(rows: list[dict]) -> pd.DataFrame:
    data = {}
    for prefix in PREFIXES:
        for position, week in enumerate((10, 11, 12, 13)):
            data[f"{prefix}_W{week}"] = [row[prefix][position] for row in rows]
    data["AON"] = [row.get("AON", 100) for row in rows]
    return pd.DataFrame(data)


def test_ratio_terminal_run_and_recharge_stop():
    rows = [{
        "DATA_MB": (10, 10, 10, 2),        # ratio 2/10 = 0.2 -> collapse
        "OG_VOICE_MIN": (5, 5, 5, 5),      # stable
        "TOTAL_SMS_COUNT": (0, 0, 0, 0),   # never active: ratio NaN, run 4
        "BUNDLE_CNT": (1, 1, 1, 0),        # terminal run 1
        "RECHARGE_AMT": (50, 50, 50, 0),   # recharge stopped
    }]
    out = features.build_features(frame(rows))
    assert out.loc[0, "FE_DATA_W13_RATIO"] == pytest.approx(0.2)
    assert np.isnan(out.loc[0, "FE_SMS_W13_RATIO"])
    assert out.loc[0, "FE_SMS_TERMINAL_ZERO_RUN"] == 4
    assert out.loc[0, "FE_BUNDLE_TERMINAL_ZERO_RUN"] == 1
    assert out.loc[0, "FE_RECHARGE_STOPPED"] == 1
    # SMS and BUNDLE are silent in W13, DATA and OG_VOICE are not -> breadth 2
    assert out.loc[0, "FE_W13_ZERO_BREADTH"] == 2
    assert out.loc[0, "FE_TERMINAL_MULTI_SERVICE"] == 1
    assert out.loc[0, "FE_ALL_CORE_ZERO_W13"] == 0


def test_all_core_zero_requires_prior_activity():
    rows = [
        {"DATA_MB": (9, 9, 9, 0), "OG_VOICE_MIN": (4, 4, 4, 0),
         "TOTAL_SMS_COUNT": (2, 2, 2, 0), "BUNDLE_CNT": (1, 1, 1, 0),
         "RECHARGE_AMT": (10, 10, 10, 0)},
        {"DATA_MB": (0, 0, 0, 0), "OG_VOICE_MIN": (0, 0, 0, 0),
         "TOTAL_SMS_COUNT": (0, 0, 0, 0), "BUNDLE_CNT": (0, 0, 0, 0),
         "RECHARGE_AMT": (0, 0, 0, 0)},
    ]
    out = features.build_features(frame(rows))
    assert out.loc[0, "FE_ALL_CORE_ZERO_W13"] == 1
    assert out.loc[1, "FE_ALL_CORE_ZERO_W13"] == 0  # never-active is not a collapse


def test_missing_service_columns_produce_nan_not_zero():
    out = features.build_features(pd.DataFrame({"AON": [100.0]}))
    assert np.isnan(out.loc[0, "FE_DATA_W13_RATIO"])
    assert np.isnan(out.loc[0, "FE_IC_VOICE_SLOPE"])
    assert np.isnan(out.loc[0, "FE_RECHARGE_STOPPED"])
    assert out.loc[0, "FE_AON_LOG"] == pytest.approx(np.log1p(100.0))
    # Cross-service features must be NaN when any core service is missing
    assert np.isnan(out.loc[0, "FE_W13_ZERO_BREADTH"])
    assert np.isnan(out.loc[0, "FE_COLLAPSE_BREADTH"])
    assert np.isnan(out.loc[0, "FE_ALL_CORE_ZERO_W13"])
    assert np.isnan(out.loc[0, "FE_TERMINAL_MULTI_SERVICE"])
    assert np.isnan(out.loc[0, "FE_MAX_TERMINAL_ZERO_RUN"])
    assert np.isnan(out.loc[0, "FE_DECLINING_SERVICES"])


def test_terminal_zero_run_counts_backwards_from_w13():
    matrix = np.array([
        [5.0, 0.0, 0.0, 0.0],  # run 3
        [5.0, 5.0, 5.0, 5.0],  # run 0
        [0.0, 5.0, 0.0, 0.0],  # run 2 (gap resets)
    ], dtype=np.float32)
    assert features.terminal_zero_run(matrix).tolist() == [3, 0, 2]


def test_monotonic_constraints_known_directions():
    names = [
        "AON",                          # raw passthrough column -> no domain assumption
        "RECHARGE_AMT_RECENT_4W",       # raw passthrough column -> no domain assumption
        "FE_DATA_W13_RATIO",            # higher ratio (less collapse) -> lower risk
        "FE_DATA_SLOPE",                # growing usage -> lower risk
        "FE_DATA_TERMINAL_ZERO_RUN",    # more consecutive silent weeks -> higher risk
        "FE_W13_ZERO_BREADTH",          # more services silent -> higher risk
        "FE_COLLAPSE_BREADTH",          # more services collapsed -> higher risk
        "FE_ALL_CORE_ZERO_W13",         # flag: 1 -> higher risk
        "FE_TERMINAL_MULTI_SERVICE",    # flag: 1 -> higher risk
        "FE_MAX_TERMINAL_ZERO_RUN",     # more silent weeks -> higher risk
        "FE_DECLINING_SERVICES",        # more declining services -> higher risk
        "FE_RECHARGE_STOPPED",          # flag: 1 -> higher risk
        "FE_AON_LOG",                   # no established domain assumption
    ]
    assert features.monotonic_constraints(names) == [
        0, 0, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
    ]


def test_usage_recharge_co_collapse_requires_both_signals():
    base_row = {
        "DATA_MB": (10, 10, 10, 1),        # collapse: ratio 0.1
        "OG_VOICE_MIN": (10, 10, 10, 1),   # collapse: ratio 0.1
        "TOTAL_SMS_COUNT": (5, 5, 5, 5),   # stable
        "BUNDLE_CNT": (2, 2, 2, 2),        # stable
    }
    rows = [
        {**base_row, "RECHARGE_AMT": (50, 50, 50, 50)},
        {**base_row, "RECHARGE_AMT": (50, 50, 50, 50)},
    ]
    df = frame(rows)
    df["RECHARGE_AMT_DROP_RATIO_2W"] = [0.8, 0.1]
    out = features.build_features(df)
    assert out.loc[0, "FE_COLLAPSE_BREADTH"] == 2
    assert out.loc[0, "FE_USAGE_RECHARGE_CO_COLLAPSE"] == 1
    assert out.loc[1, "FE_USAGE_RECHARGE_CO_COLLAPSE"] == 0


def test_usage_recharge_co_collapse_requires_breadth_of_two():
    rows = [{
        "DATA_MB": (10, 10, 10, 1),          # collapse
        "OG_VOICE_MIN": (10, 10, 10, 10),    # stable
        "TOTAL_SMS_COUNT": (5, 5, 5, 5),     # stable
        "BUNDLE_CNT": (2, 2, 2, 2),          # stable
        "RECHARGE_AMT": (50, 50, 50, 50),
    }]
    df = frame(rows)
    df["RECHARGE_AMT_DROP_RATIO_2W"] = [0.9]
    out = features.build_features(df)
    assert out.loc[0, "FE_COLLAPSE_BREADTH"] == 1
    assert out.loc[0, "FE_USAGE_RECHARGE_CO_COLLAPSE"] == 0


def test_usage_recharge_co_collapse_nan_when_column_missing():
    out = features.build_features(pd.DataFrame({"AON": [100.0]}))
    assert np.isnan(out.loc[0, "FE_USAGE_RECHARGE_CO_COLLAPSE"])


def test_usage_recharge_co_collapse_nan_when_drop_ratio_missing_but_core_present():
    rows = [{
        "DATA_MB": (10, 10, 10, 1), "OG_VOICE_MIN": (10, 10, 10, 1),
        "TOTAL_SMS_COUNT": (5, 5, 5, 5), "BUNDLE_CNT": (2, 2, 2, 2),
        "RECHARGE_AMT": (50, 50, 50, 50),
    }]
    out = features.build_features(frame(rows))  # frame() never adds RECHARGE_AMT_DROP_RATIO_2W
    assert out.loc[0, "FE_COLLAPSE_BREADTH"] == 2
    assert np.isnan(out.loc[0, "FE_USAGE_RECHARGE_CO_COLLAPSE"])


def test_low_engagement_service_count_counts_zero_columns():
    df = pd.DataFrame({
        "DATA_ACTIVE_DAYS_RECENT_4W": [0, 5],
        "BUNDLE_ACTIVE_DAYS_RECENT_4W": [0, 0],
        "TOTAL_VOICE_ACTIVE_WEEKS_RECENT_4W": [2, 0],
        "TOTAL_SMS_ACTIVE_WEEKS_RECENT_4W": [0, 3],
        "RECHARGE_ACTIVE_WEEKS_4W": [1, 4],
    })
    out = features.build_features(df)
    assert out.loc[0, "FE_LOW_ENGAGEMENT_SERVICE_COUNT"] == 3
    assert out.loc[1, "FE_LOW_ENGAGEMENT_SERVICE_COUNT"] == 2


def test_low_engagement_service_count_nan_when_any_column_missing():
    df = pd.DataFrame({
        "DATA_ACTIVE_DAYS_RECENT_4W": [0],
        "BUNDLE_ACTIVE_DAYS_RECENT_4W": [0],
        "TOTAL_VOICE_ACTIVE_WEEKS_RECENT_4W": [0],
        "TOTAL_SMS_ACTIVE_WEEKS_RECENT_4W": [0],
    })
    out = features.build_features(df)
    assert np.isnan(out.loc[0, "FE_LOW_ENGAGEMENT_SERVICE_COUNT"])


def test_revenue_at_risk_multiplies_revenue_by_collapse_breadth():
    rows = [{
        "DATA_MB": (10, 10, 10, 1), "OG_VOICE_MIN": (10, 10, 10, 1),
        "TOTAL_SMS_COUNT": (5, 5, 5, 5), "BUNDLE_CNT": (2, 2, 2, 2),
        "RECHARGE_AMT": (50, 50, 50, 50),
    }]
    df = frame(rows)
    df["TOTAL_REVENUE_RECENT_4W"] = [200.0]
    out = features.build_features(df)
    assert out.loc[0, "FE_COLLAPSE_BREADTH"] == 2
    assert out.loc[0, "FE_REVENUE_AT_RISK"] == pytest.approx(400.0)


def test_revenue_at_risk_nan_when_revenue_column_missing():
    out = features.build_features(pd.DataFrame({"AON": [100.0]}))
    assert np.isnan(out.loc[0, "FE_REVENUE_AT_RISK"])


def test_revenue_at_risk_nan_when_core_missing_even_with_revenue_present():
    df = pd.DataFrame({"AON": [100.0], "TOTAL_REVENUE_RECENT_4W": [500.0]})
    out = features.build_features(df)
    assert np.isnan(out.loc[0, "FE_REVENUE_AT_RISK"])


def test_monotonic_constraints_include_new_composite_features():
    names = [
        "FE_USAGE_RECHARGE_CO_COLLAPSE",       # flag: 1 -> higher risk
        "FE_LOW_ENGAGEMENT_SERVICE_COUNT",     # more low-engagement services -> higher risk
        "FE_REVENUE_AT_RISK",                  # ambiguous -- no established domain assumption
    ]
    assert features.monotonic_constraints(names) == [1, 1, 0]


def test_partial_missing_core_service_produces_nan_cross_features():
    # DATA_MB, OG_VOICE_MIN, TOTAL_SMS_COUNT, RECHARGE_AMT present
    # but BUNDLE_CNT (core service) is missing
    df = pd.DataFrame({
        "DATA_MB_W10": [10.0],
        "DATA_MB_W11": [10.0],
        "DATA_MB_W12": [10.0],
        "DATA_MB_W13": [5.0],
        "OG_VOICE_MIN_W10": [5.0],
        "OG_VOICE_MIN_W11": [5.0],
        "OG_VOICE_MIN_W12": [5.0],
        "OG_VOICE_MIN_W13": [5.0],
        "TOTAL_SMS_COUNT_W10": [2.0],
        "TOTAL_SMS_COUNT_W11": [2.0],
        "TOTAL_SMS_COUNT_W12": [2.0],
        "TOTAL_SMS_COUNT_W13": [2.0],
        "RECHARGE_AMT_W10": [50.0],
        "RECHARGE_AMT_W11": [50.0],
        "RECHARGE_AMT_W12": [50.0],
        "RECHARGE_AMT_W13": [50.0],
    })
    out = features.build_features(df)
    # Per-service columns for present services should be computed
    assert out.loc[0, "FE_DATA_W13_RATIO"] == pytest.approx(0.5)
    assert out.loc[0, "FE_OG_VOICE_W13_RATIO"] == pytest.approx(1.0)
    assert out.loc[0, "FE_SMS_W13_RATIO"] == pytest.approx(1.0)
    # BUNDLE missing -> its per-service columns are NaN
    assert np.isnan(out.loc[0, "FE_BUNDLE_W13_RATIO"])
    assert np.isnan(out.loc[0, "FE_BUNDLE_SLOPE"])
    assert np.isnan(out.loc[0, "FE_BUNDLE_TERMINAL_ZERO_RUN"])
    # Cross-service features must all be NaN (core service missing)
    assert np.isnan(out.loc[0, "FE_W13_ZERO_BREADTH"])
    assert np.isnan(out.loc[0, "FE_COLLAPSE_BREADTH"])
    assert np.isnan(out.loc[0, "FE_ALL_CORE_ZERO_W13"])
    assert np.isnan(out.loc[0, "FE_TERMINAL_MULTI_SERVICE"])
    assert np.isnan(out.loc[0, "FE_MAX_TERMINAL_ZERO_RUN"])
    assert np.isnan(out.loc[0, "FE_DECLINING_SERVICES"])
