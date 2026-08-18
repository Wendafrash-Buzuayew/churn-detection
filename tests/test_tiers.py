import numpy as np
import pandas as pd

from churn_ranker import tiers


def test_thresholds_and_assignment_volumes():
    scores = np.linspace(0.0, 1.0, 100)
    thresholds = tiers.tier_thresholds(scores, (("T1", 0.01), ("T2", 0.05)))
    labels = tiers.assign_tiers(scores, thresholds)
    assert (labels == "T1").sum() == 1        # top 1% of 100 evenly spaced scores
    assert (labels == "T2").sum() == 4        # next 4% (5% cumulative minus T1)
    assert labels[0] == tiers.STABLE


def test_tier_order_is_strictest_first():
    thresholds = [("T1", 0.9), ("T2", 0.5)]
    labels = tiers.assign_tiers(np.array([0.95, 0.7, 0.1]), thresholds)
    assert labels.tolist() == ["T1", "T2", tiers.STABLE]


def test_reason_priority_order():
    derived = pd.DataFrame({
        "FE_ALL_CORE_ZERO_W13": [1, 0, 0, 0, 0],
        "FE_TERMINAL_MULTI_SERVICE": [1, 1, 0, 0, 0],
        "FE_RECHARGE_STOPPED": [1, 1, 1, 0, 0],
        "FE_DATA_W13_RATIO": [0.0, 0.0, 1.0, 0.1, 1.0],
        "FE_OG_VOICE_W13_RATIO": [0.0, 0.0, 1.0, 1.0, 1.0],
        "FE_DECLINING_SERVICES": [3, 3, 0, 0, 0],
    })
    tier_labels = np.array(["TIER_1_IMMINENT"] * 5)
    reasons = tiers.reason_codes(derived, tier_labels)
    assert reasons.tolist() == [
        "all_services_silent_last_week",
        "multi_service_collapse",
        "recharge_stopped",
        "data_usage_collapse",
        "model_pattern",
    ]


def test_stable_rows_get_empty_reason():
    derived = pd.DataFrame({
        "FE_ALL_CORE_ZERO_W13": [1],
        "FE_TERMINAL_MULTI_SERVICE": [1],
        "FE_RECHARGE_STOPPED": [1],
        "FE_DATA_W13_RATIO": [0.0],
        "FE_OG_VOICE_W13_RATIO": [0.0],
        "FE_DECLINING_SERVICES": [3],
    })
    reasons = tiers.reason_codes(derived, np.array([tiers.STABLE]))
    assert reasons.tolist() == [""]
