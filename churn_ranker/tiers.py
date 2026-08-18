"""Capacity-based tier assignment and priority-ordered reason codes."""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_TIERS = (
    ("TIER_1_IMMINENT", 0.01),
    ("TIER_2_HIGH_RISK", 0.05),
    ("TIER_3_WATCHLIST", 0.15),
)
STABLE = "STABLE"
COLLAPSE_RATIO = 0.2
GRADUAL_DECLINE_SERVICES = 2


def tier_thresholds(scores, tier_spec=DEFAULT_TIERS) -> list[tuple[str, float]]:
    """Score quantile per cumulative top fraction, fixed at training time."""
    s = np.asarray(scores, dtype=float)
    return [
        (name, float(np.quantile(s, 1.0 - top_fraction)))
        for name, top_fraction in tier_spec
    ]


def assign_tiers(scores, thresholds) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    result = np.full(len(s), STABLE, dtype=object)
    assigned = np.zeros(len(s), dtype=bool)
    for name, threshold in thresholds:  # ordered strictest (highest) first
        selected = ~assigned & (s >= threshold)
        result[selected] = name
        assigned |= selected
    return result.astype(str)


def _flag(derived: pd.DataFrame, column: str) -> np.ndarray:
    if column not in derived.columns:
        return np.zeros(len(derived), dtype=bool)
    values = pd.to_numeric(derived[column], errors="coerce").fillna(0).to_numpy(dtype=float)
    return values >= 1


def _ratio(derived: pd.DataFrame, column: str) -> np.ndarray:
    if column not in derived.columns:
        return np.full(len(derived), 1.0)
    values = pd.to_numeric(derived[column], errors="coerce").to_numpy(dtype=float)
    return np.nan_to_num(values, nan=1.0)


def reason_codes(derived: pd.DataFrame, tier_labels: np.ndarray) -> np.ndarray:
    conditions = [
        _flag(derived, "FE_ALL_CORE_ZERO_W13"),
        _flag(derived, "FE_TERMINAL_MULTI_SERVICE"),
        _flag(derived, "FE_RECHARGE_STOPPED"),
        _ratio(derived, "FE_DATA_W13_RATIO") <= COLLAPSE_RATIO,
        _ratio(derived, "FE_OG_VOICE_W13_RATIO") <= COLLAPSE_RATIO,
        _ratio(derived, "FE_DECLINING_SERVICES") >= GRADUAL_DECLINE_SERVICES,
    ]
    choices = [
        "all_services_silent_last_week",
        "multi_service_collapse",
        "recharge_stopped",
        "data_usage_collapse",
        "voice_usage_collapse",
        "gradual_decline",
    ]
    reasons = np.select(conditions, choices, default="model_pattern")
    return np.where(np.asarray(tier_labels) == STABLE, "", reasons).astype(str)
