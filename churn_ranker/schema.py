"""Canonical schema resolution across churn CSVs with differing column names."""
from __future__ import annotations

import pandas as pd

TARGET = "LABEL_CHURN_90D"

EXCLUDE_EXACT = {"SNAPSHOT_DATE", "DATASET_TYPE", "LAST_RECHARGE_RECENCY_BAND"}
EXCLUDE_PREFIXES = ("MSISDN", "LABEL_", "LAST_RECHARGE_DATE")

# canonical name -> aliases seen in other files (renamed only when canonical absent)
ALIASES = {
    "RECHARGE_AMT_RECENT_4W": ("RECHARGE_AMT_TOTAL_4W",),
    "RECHARGE_CNT_RECENT_4W": ("RECHARGE_CNT_TOTAL_4W",),
    "RECHARGE_AVG_WEEKLY_AMT_4W": ("RECHARGE_AMT_AVG_WEEKLY_4W",),
    "DAYS_SINCE_LAST_RECHARGE": ("DAYS_SINCE_LAST_RECHARGE_4W",),
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).upper().strip() for c in out.columns]
    return out


def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)
    renames: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            if alias in out.columns:
                renames[alias] = canonical
                break
    return out.rename(columns=renames)


def is_excluded(column: str) -> bool:
    if column == TARGET or column in EXCLUDE_EXACT:
        return True
    return any(column.startswith(prefix) for prefix in EXCLUDE_PREFIXES)


def feature_columns(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include="number").columns
    return [c for c in numeric if not is_excluded(c)]
