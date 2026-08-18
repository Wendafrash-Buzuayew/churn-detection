"""Derived churn signals computable on every input file's shared W10-W13 schema."""
from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS = (10, 11, 12, 13)
SERVICES = {
    "DATA": "DATA_MB",
    "OG_VOICE": "OG_VOICE_MIN",
    "IC_VOICE": "IC_VOICE_MIN",
    "SMS": "TOTAL_SMS_COUNT",
    "BUNDLE": "BUNDLE_CNT",
    "RECHARGE_AMT": "RECHARGE_AMT",
    "RECHARGE_CNT": "RECHARGE_CNT",
}
CORE_SERVICES = ("DATA", "OG_VOICE", "SMS", "BUNDLE")
COLLAPSE_RATIO = 0.2
DECLINE_SLOPE = -0.3


def week_matrix(df: pd.DataFrame, prefix: str) -> np.ndarray | None:
    columns = [f"{prefix}_W{week}" for week in WEEKS]
    if not all(c in df.columns for c in columns):
        return None
    values = [
        pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        for c in columns
    ]
    return np.column_stack(values)


def terminal_zero_run(matrix: np.ndarray) -> np.ndarray:
    """Consecutive zero weeks counted backwards from W13 (max 4)."""
    zeros = matrix <= 0
    run = np.zeros(len(matrix), dtype=np.int8)
    still_zero = np.ones(len(matrix), dtype=bool)
    for position in (3, 2, 1, 0):
        still_zero &= zeros[:, position]
        run += still_zero
    return run


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    n = len(df)
    nan_column = np.full(n, np.nan, dtype=np.float32)
    core_w13_zero, core_collapse, core_runs = [], [], []
    core_baselines, core_slopes = [], []
    for name, prefix in SERVICES.items():
        matrix = week_matrix(df, prefix)
        if matrix is None:
            out[f"FE_{name}_W13_RATIO"] = nan_column
            out[f"FE_{name}_SLOPE"] = nan_column
            out[f"FE_{name}_TERMINAL_ZERO_RUN"] = nan_column
            if name in CORE_SERVICES:
                core_w13_zero.append(np.zeros(n, dtype=bool))
                core_collapse.append(np.zeros(n, dtype=bool))
                core_runs.append(np.zeros(n, dtype=np.int8))
                core_baselines.append(np.zeros(n, dtype=np.float32))
                core_slopes.append(np.zeros(n, dtype=np.float32))
            continue
        baseline = matrix[:, :3].mean(axis=1)
        safe_baseline = np.where(baseline > 0, baseline, 1.0)
        ratio = np.where(baseline > 0, matrix[:, 3] / safe_baseline, np.nan).astype(np.float32)
        slope = ((matrix[:, 3] - matrix[:, 0]) / (np.abs(matrix[:, 0]) + 1.0)).astype(np.float32)
        run = terminal_zero_run(matrix)
        out[f"FE_{name}_W13_RATIO"] = ratio
        out[f"FE_{name}_SLOPE"] = slope
        out[f"FE_{name}_TERMINAL_ZERO_RUN"] = run
        if name in CORE_SERVICES:
            core_w13_zero.append(matrix[:, 3] <= 0)
            core_collapse.append((baseline > 0) & (np.nan_to_num(ratio, nan=1.0) <= COLLAPSE_RATIO))
            core_runs.append(run)
            core_baselines.append(baseline)
            core_slopes.append(slope)
    zero_breadth = np.column_stack(core_w13_zero).sum(axis=1).astype(np.int8)
    baseline_total = np.column_stack(core_baselines).sum(axis=1)
    out["FE_W13_ZERO_BREADTH"] = zero_breadth
    out["FE_COLLAPSE_BREADTH"] = np.column_stack(core_collapse).sum(axis=1).astype(np.int8)
    out["FE_ALL_CORE_ZERO_W13"] = (
        (zero_breadth == len(CORE_SERVICES)) & (baseline_total > 0)
    ).astype(np.int8)
    out["FE_TERMINAL_MULTI_SERVICE"] = (
        (zero_breadth >= 2) & (baseline_total > 0)
    ).astype(np.int8)
    out["FE_MAX_TERMINAL_ZERO_RUN"] = np.maximum.reduce(core_runs)
    out["FE_DECLINING_SERVICES"] = (
        np.column_stack(core_slopes) < DECLINE_SLOPE
    ).sum(axis=1).astype(np.int8)
    recharge = week_matrix(df, SERVICES["RECHARGE_AMT"])
    if recharge is None:
        out["FE_RECHARGE_STOPPED"] = nan_column
    else:
        recharge_baseline = recharge[:, :3].mean(axis=1)
        out["FE_RECHARGE_STOPPED"] = (
            (recharge_baseline > 0) & (recharge[:, 3] <= 0)
        ).astype(np.int8)
    if "AON" in df.columns:
        aon = pd.to_numeric(df["AON"], errors="coerce")
        out["FE_AON_LOG"] = np.log1p(aon.clip(lower=0)).astype(np.float32)
    else:
        out["FE_AON_LOG"] = nan_column
    return out
