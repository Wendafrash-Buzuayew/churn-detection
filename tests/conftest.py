import numpy as np
import pandas as pd
import pytest

WEEK_PREFIXES = (
    "DATA_MB", "OG_VOICE_MIN", "IC_VOICE_MIN", "TOTAL_SMS_COUNT",
    "BUNDLE_CNT", "RECHARGE_AMT", "RECHARGE_CNT",
)
CHURN_WEEK_MULTIPLIERS = (1.0, 0.8, 0.15, 0.0)


def make_synthetic(n: int = 600, churn_frac: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """Labeled frame in the canonical schema; churners collapse over W10-W13."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[: int(n * churn_frac)] = 1
    rng.shuffle(y)
    # Generate MSISDNs offset by seed for cross-dataset uniqueness
    start = seed * 10000
    df = pd.DataFrame({
        "MSISDN": [f"2519{start + i:08d}" for i in range(n)],
        "SNAPSHOT_DATE": "2026-01-15",
        "DATASET_TYPE": "TRAIN",
        "AON": rng.integers(91, 1200, n),
        "LABEL_CHURN_90D": y,
    })
    for prefix in WEEK_PREFIXES:
        base = rng.lognormal(3.0, 1.0, n)
        for position, week in enumerate((10, 11, 12, 13)):
            noise = rng.uniform(0.7, 1.3, n)
            decay = np.where(y == 1, CHURN_WEEK_MULTIPLIERS[position], 1.0)
            df[f"{prefix}_W{week}"] = base * noise * decay
    return df


@pytest.fixture
def synthetic() -> pd.DataFrame:
    return make_synthetic()
