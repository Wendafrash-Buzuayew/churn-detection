# Churn Risk Ranking Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the binary churn classifier with a calibrated risk-ranking engine that trains on the 521K-row Feb dataset, validates out-of-sample on the Jan-15 snapshot, and outputs capacity-sized action tiers with measured precision and reason codes.

**Architecture:** New `churn_ranker/` package with five focused modules (schema harmonization, derived features, gradient-boosted ranking model with calibration, ranking evaluation, tier/reason assignment) plus a CLI. Old scripts (`unified_churn_engine.py`, `churn_rules*.py`) stay untouched as reference.

**Tech Stack:** Python via `./venv/Scripts/python.exe` — pandas 3.0.3, numpy 2.4.6, scikit-learn 1.5.2 (HistGradientBoostingClassifier), joblib, pytest (dev-only, must be installed in Task 1).

**Spec:** `docs/superpowers/specs/2026-08-18-churn-risk-ranking-design.md`

## Global Constraints

- Interpreter is ALWAYS `./venv/Scripts/python.exe` (Windows venv; plain `python` may resolve elsewhere).
- pandas is **3.0.3**: no chained assignment, pass `numeric_only=True` where reductions could see non-numeric columns, never rely on silent downcasting.
- scikit-learn pinned `>=1.2.0,<1.6.0` (installed 1.5.2). No new runtime dependencies; `pytest>=8.0` is the only addition (dev section of `requirements.txt`).
- Target column is `LABEL_CHURN_90D`. Never let `MSISDN*`, `LABEL_*`, `SNAPSHOT_DATE`, `DATASET_TYPE`, `LAST_RECHARGE_DATE*`, or `LAST_RECHARGE_RECENCY_BAND` into the model matrix.
- `Sample_data_full_feature.csv` (7,008 rows, 38 churners) is a smoke-test fixture ONLY. Model quality conclusions come from `Feb1_Train_with_recharg.csv` (train) and `March_validation_with_recharg.csv` (validation, overlap-filtered).
- All generated outputs go to `churn_ranker_outputs/` (gitignored in Task 7). Never commit CSVs or joblib artifacts.
- Do not modify `unified_churn_engine.py`, `churn_rules*.py`, `churn_pipeline*.py`, or `tests/test_unified_churn_engine.py`.
- Run tests as: `./venv/Scripts/python.exe -m pytest tests/<file> -v` (never bare `pytest`).

---

### Task 1: Package scaffold + schema module

**Files:**
- Create: `churn_ranker/__init__.py`
- Create: `churn_ranker/schema.py`
- Create: `tests/conftest.py`
- Test: `tests/test_schema.py`
- Modify: `requirements.txt` (append dev section)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `schema.TARGET: str`, `schema.normalize_columns(df: pd.DataFrame) -> pd.DataFrame`, `schema.to_canonical(df: pd.DataFrame) -> pd.DataFrame`, `schema.is_excluded(column: str) -> bool`, `schema.feature_columns(df: pd.DataFrame) -> list[str]`. Also the pytest fixture `synthetic` (a labeled DataFrame) used by Tasks 5–6.

- [ ] **Step 1: Install pytest into the venv and record it**

Run: `./venv/Scripts/python.exe -m pip install "pytest>=8.0"`

Append to `requirements.txt`:

```text

# Development / testing
pytest>=8.0
```

- [ ] **Step 2: Create the package and the shared synthetic fixture**

`churn_ranker/__init__.py`:

```python
"""Churn risk ranking engine: calibrated scores, capacity tiers, reason codes."""
```

`tests/conftest.py`:

```python
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
    df = pd.DataFrame({
        "MSISDN": [f"2519{i:08d}" for i in range(n)],
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
```

- [ ] **Step 3: Write the failing schema tests**

`tests/test_schema.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_ranker.schema'` (or ImportError).

- [ ] **Step 5: Implement the schema module**

`churn_ranker/schema.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_schema.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add churn_ranker/__init__.py churn_ranker/schema.py tests/conftest.py tests/test_schema.py requirements.txt
git commit -m "feat: churn_ranker package scaffold with canonical schema resolution"
```

---

### Task 2: Derived feature builder

**Files:**
- Create: `churn_ranker/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: a canonical DataFrame (output of `schema.to_canonical`).
- Produces: `features.build_features(df: pd.DataFrame) -> pd.DataFrame` returning ONLY derived columns, all prefixed `FE_`: per service `FE_{NAME}_W13_RATIO`, `FE_{NAME}_SLOPE`, `FE_{NAME}_TERMINAL_ZERO_RUN` for NAME in `DATA, OG_VOICE, IC_VOICE, SMS, BUNDLE, RECHARGE_AMT, RECHARGE_CNT`; cross-service `FE_W13_ZERO_BREADTH`, `FE_COLLAPSE_BREADTH`, `FE_ALL_CORE_ZERO_W13`, `FE_TERMINAL_MULTI_SERVICE`, `FE_MAX_TERMINAL_ZERO_RUN`, `FE_DECLINING_SERVICES`, `FE_RECHARGE_STOPPED`, `FE_AON_LOG`. Missing input columns yield NaN feature columns (never zeros). Also `features.week_matrix(df, prefix) -> np.ndarray | None` and `features.terminal_zero_run(matrix) -> np.ndarray`.

- [ ] **Step 1: Write the failing feature tests**

`tests/test_features.py`:

```python
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


def test_terminal_zero_run_counts_backwards_from_w13():
    matrix = np.array([
        [5.0, 0.0, 0.0, 0.0],  # run 3
        [5.0, 5.0, 5.0, 5.0],  # run 0
        [0.0, 5.0, 0.0, 0.0],  # run 2 (gap resets)
    ], dtype=np.float32)
    assert features.terminal_zero_run(matrix).tolist() == [3, 0, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_ranker.features'`.

- [ ] **Step 3: Implement the feature builder**

`churn_ranker/features.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_features.py tests/test_schema.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add churn_ranker/features.py tests/test_features.py
git commit -m "feat: vectorized derived churn signals with NaN-safe missing-column handling"
```

---

### Task 3: Ranking evaluation module

**Files:**
- Create: `churn_ranker/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: numpy-compatible arrays of labels and scores.
- Produces: `evaluation.ranking_metrics(y_true, scores) -> dict` with keys `n, positives, base_rate, roc_auc, pr_auc` (AUCs `None` if one class); `evaluation.lift_table(y_true, scores, top_fractions=evaluation.TOP_FRACTIONS) -> pd.DataFrame` with columns `top_fraction, contacted, churners_caught, precision, recall, lift`; constant `evaluation.TOP_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20)`.

- [ ] **Step 1: Write the failing evaluation tests**

`tests/test_evaluation.py`:

```python
import numpy as np
import pytest

from churn_ranker import evaluation


def test_lift_table_hand_computed():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    table = evaluation.lift_table(y, scores, top_fractions=(0.1, 0.3))
    top10 = table.iloc[0]
    assert top10["contacted"] == 1
    assert top10["churners_caught"] == 1
    assert top10["precision"] == pytest.approx(1.0)
    assert top10["recall"] == pytest.approx(0.5)
    assert top10["lift"] == pytest.approx(5.0)  # precision 1.0 / base rate 0.2
    top30 = table.iloc[1]
    assert top30["contacted"] == 3
    assert top30["churners_caught"] == 2
    assert top30["recall"] == pytest.approx(1.0)


def test_ranking_metrics_keys_and_base_rate():
    y = np.array([1, 0, 0, 0])
    scores = np.array([0.9, 0.2, 0.1, 0.4])
    metrics = evaluation.ranking_metrics(y, scores)
    assert metrics["n"] == 4
    assert metrics["positives"] == 1
    assert metrics["base_rate"] == pytest.approx(0.25)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert 0.0 < metrics["pr_auc"] <= 1.0


def test_single_class_returns_none_aucs():
    metrics = evaluation.ranking_metrics(np.zeros(5, dtype=int), np.linspace(0, 1, 5))
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_ranker.evaluation'`.

- [ ] **Step 3: Implement the evaluation module**

`churn_ranker/evaluation.py`:

```python
"""Ranking-quality metrics: AUCs and capacity lift tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TOP_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20)


def ranking_metrics(y_true, scores) -> dict:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    out = {"n": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean())}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def lift_table(y_true, scores, top_fractions=TOP_FRACTIONS) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="stable")
    cumulative_positives = np.cumsum(y[order])
    total_positives = max(int(y.sum()), 1)
    base_rate = max(float(y.mean()), 1e-12)
    rows = []
    for fraction in top_fractions:
        contacted = max(1, int(round(len(y) * fraction)))
        caught = int(cumulative_positives[contacted - 1])
        precision = caught / contacted
        rows.append({
            "top_fraction": fraction,
            "contacted": contacted,
            "churners_caught": caught,
            "precision": precision,
            "recall": caught / total_positives,
            "lift": precision / base_rate,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_evaluation.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add churn_ranker/evaluation.py tests/test_evaluation.py
git commit -m "feat: ranking metrics and capacity lift table"
```

---

### Task 4: Tier assignment and reason codes

**Files:**
- Create: `churn_ranker/tiers.py`
- Test: `tests/test_tiers.py`

**Interfaces:**
- Consumes: calibrated score arrays; the derived-feature DataFrame from `features.build_features` (columns `FE_ALL_CORE_ZERO_W13`, `FE_TERMINAL_MULTI_SERVICE`, `FE_RECHARGE_STOPPED`, `FE_DATA_W13_RATIO`, `FE_OG_VOICE_W13_RATIO`, `FE_DECLINING_SERVICES`).
- Produces: `tiers.DEFAULT_TIERS = (("TIER_1_IMMINENT", 0.01), ("TIER_2_HIGH_RISK", 0.05), ("TIER_3_WATCHLIST", 0.15))`, `tiers.STABLE = "STABLE"`, `tiers.tier_thresholds(scores, tier_spec=DEFAULT_TIERS) -> list[tuple[str, float]]`, `tiers.assign_tiers(scores, thresholds) -> np.ndarray` (str labels), `tiers.reason_codes(derived: pd.DataFrame, tier_labels: np.ndarray) -> np.ndarray` (str; empty string for STABLE rows).

- [ ] **Step 1: Write the failing tier tests**

`tests/test_tiers.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_tiers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_ranker.tiers'`.

- [ ] **Step 3: Implement the tiers module**

`churn_ranker/tiers.py`:

```python
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
```

Note: `_ratio` is reused for `FE_DECLINING_SERVICES` (a count) because it has the same
"numeric with NaN->neutral" semantics; NaN becomes 1.0 which is below the 2-service
gate, so missing data never claims "gradual_decline".

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_tiers.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add churn_ranker/tiers.py tests/test_tiers.py
git commit -m "feat: capacity tier thresholds and priority reason codes"
```

---

### Task 5: ChurnRanker model (CV, calibration, persistence)

**Files:**
- Create: `churn_ranker/modeling.py`
- Test: `tests/test_modeling.py`

**Interfaces:**
- Consumes: `schema.to_canonical`, `schema.feature_columns`, `schema.TARGET`, `features.build_features`, `evaluation.ranking_metrics`, `evaluation.lift_table`, `tiers.tier_thresholds`, `tiers.assign_tiers`, `tiers.reason_codes`, `tiers.DEFAULT_TIERS` — exactly as defined in Tasks 1–4.
- Produces: `RankerConfig` dataclass (fields `target, cv_folds, random_state, max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, tier_spec`); `ChurnRanker` with `fit(df) -> dict`, `predict_proba(df) -> np.ndarray`, `score(df) -> pd.DataFrame` (adds `churn_probability`, `risk_tier`, `reason_code`), `save(path)`, `ChurnRanker.load(path)`, attributes `feature_names_: list[str]`, `tier_thresholds_: list[tuple[str, float]]`, `training_summary_: dict`.

- [ ] **Step 1: Write the failing modeling tests**

`tests/test_modeling.py`:

```python
import numpy as np
import pytest

from churn_ranker.modeling import ChurnRanker, RankerConfig

FAST = RankerConfig(cv_folds=3, max_iter=60)


def test_fit_learns_synthetic_collapse(synthetic):
    ranker = ChurnRanker(FAST)
    summary = ranker.fit(synthetic)
    assert summary["oof_metrics"]["roc_auc"] > 0.85
    assert summary["n_churners"] == int(synthetic["LABEL_CHURN_90D"].sum())
    assert len(summary["tier_thresholds"]) == 3


def test_leakage_columns_never_enter_features(synthetic):
    ranker = ChurnRanker(FAST)
    ranker.fit(synthetic)
    for name in ranker.feature_names_:
        assert not name.startswith(("MSISDN", "LABEL_"))
        assert name not in ("SNAPSHOT_DATE", "DATASET_TYPE")


def test_score_output_columns_and_ranges(synthetic):
    ranker = ChurnRanker(FAST)
    ranker.fit(synthetic)
    scored = ranker.score(synthetic)
    assert {"churn_probability", "risk_tier", "reason_code"} <= set(scored.columns)
    assert scored["churn_probability"].between(0, 1).all()
    assert set(scored["risk_tier"]) <= {
        "TIER_1_IMMINENT", "TIER_2_HIGH_RISK", "TIER_3_WATCHLIST", "STABLE",
    }
    stable = scored["risk_tier"] == "STABLE"
    assert (scored.loc[stable, "reason_code"] == "").all()
    assert (scored.loc[~stable, "reason_code"] != "").all()


def test_scores_unlabeled_data(synthetic):
    ranker = ChurnRanker(FAST)
    ranker.fit(synthetic)
    unlabeled = synthetic.drop(columns=["LABEL_CHURN_90D"])
    scored = ranker.score(unlabeled)
    assert len(scored) == len(unlabeled)


def test_save_load_roundtrip(tmp_path, synthetic):
    ranker = ChurnRanker(FAST)
    ranker.fit(synthetic)
    path = tmp_path / "ranker.joblib"
    ranker.save(path)
    loaded = ChurnRanker.load(path)
    np.testing.assert_allclose(
        ranker.predict_proba(synthetic), loaded.predict_proba(synthetic)
    )


def test_fit_without_target_raises(synthetic):
    with pytest.raises(ValueError):
        ChurnRanker(FAST).fit(synthetic.drop(columns=["LABEL_CHURN_90D"]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_modeling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_ranker.modeling'`.

- [ ] **Step 3: Implement the model**

`churn_ranker/modeling.py`:

```python
"""Calibrated churn risk ranker with training-time capacity-tier thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from churn_ranker import evaluation, features, schema, tiers


@dataclass
class RankerConfig:
    target: str = schema.TARGET
    cv_folds: int = 5
    random_state: int = 42
    max_iter: int = 300
    learning_rate: float = 0.06
    max_leaf_nodes: int = 63
    min_samples_leaf: int = 60
    tier_spec: tuple = tiers.DEFAULT_TIERS


class ChurnRanker:
    def __init__(self, config: RankerConfig | None = None):
        self.config = config or RankerConfig()
        self.model: HistGradientBoostingClassifier | None = None
        self.calibrator: LogisticRegression | None = None
        self.feature_names_: list[str] = []
        self.tier_thresholds_: list[tuple[str, float]] = []
        self.training_summary_: dict[str, Any] = {}

    def _matrix(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        canonical = schema.to_canonical(df)
        derived = features.build_features(canonical)
        base = canonical[schema.feature_columns(canonical)]
        full = pd.concat([base, derived], axis=1)
        if fit:
            self.feature_names_ = list(full.columns)
        full = full.reindex(columns=self.feature_names_)
        return full.to_numpy(dtype=np.float32)

    def _base_model(self) -> HistGradientBoostingClassifier:
        c = self.config
        return HistGradientBoostingClassifier(
            max_iter=c.max_iter,
            learning_rate=c.learning_rate,
            max_leaf_nodes=c.max_leaf_nodes,
            min_samples_leaf=c.min_samples_leaf,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=c.random_state,
        )

    def fit(self, df: pd.DataFrame) -> dict[str, Any]:
        canonical = schema.to_canonical(df)
        if self.config.target not in canonical.columns:
            raise ValueError(f"Training data must contain {self.config.target}")
        y = (
            pd.to_numeric(canonical[self.config.target], errors="coerce")
            .fillna(0).astype(int).to_numpy()
        )
        if len(np.unique(y)) < 2:
            raise ValueError("Training data needs both churn classes")
        X = self._matrix(df, fit=True)
        folds = min(self.config.cv_folds, int(y.sum()), int(len(y) - y.sum()))
        if folds < 2:
            raise ValueError("Not enough minority examples for cross-validation")
        oof = np.zeros(len(y))
        splitter = StratifiedKFold(folds, shuffle=True, random_state=self.config.random_state)
        for train_idx, valid_idx in splitter.split(X, y):
            fold_model = self._base_model()
            fold_model.fit(X[train_idx], y[train_idx])
            oof[valid_idx] = fold_model.predict_proba(X[valid_idx])[:, 1]
        self.calibrator = LogisticRegression(random_state=self.config.random_state)
        self.calibrator.fit(oof.reshape(-1, 1), y)
        calibrated = self.calibrator.predict_proba(oof.reshape(-1, 1))[:, 1]
        self.tier_thresholds_ = tiers.tier_thresholds(calibrated, self.config.tier_spec)
        self.model = self._base_model()
        self.model.fit(X, y)
        self.training_summary_ = {
            "n_rows": int(len(y)),
            "n_churners": int(y.sum()),
            "n_features": len(self.feature_names_),
            "cv_folds": folds,
            "tier_thresholds": [[name, float(t)] for name, t in self.tier_thresholds_],
            "oof_metrics": evaluation.ranking_metrics(y, calibrated),
            "oof_lift_table": evaluation.lift_table(y, calibrated).to_dict(orient="records"),
        }
        return self.training_summary_

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.calibrator is None:
            raise RuntimeError("ChurnRanker is not fitted")
        raw = self.model.predict_proba(self._matrix(df))[:, 1]
        return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        probability = self.predict_proba(df)
        derived = features.build_features(schema.to_canonical(df))
        result = df.copy()
        result["churn_probability"] = probability
        tier_labels = tiers.assign_tiers(probability, self.tier_thresholds_)
        result["risk_tier"] = tier_labels
        result["reason_code"] = tiers.reason_codes(derived, tier_labels)
        return result

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ChurnRanker":
        return joblib.load(path)
```

- [ ] **Step 4: Run the full test suite so far**

Run: `./venv/Scripts/python.exe -m pytest tests/test_schema.py tests/test_features.py tests/test_evaluation.py tests/test_tiers.py tests/test_modeling.py -v`
Expected: all pass. (`test_fit_learns_synthetic_collapse` may take ~10–30s.)

- [ ] **Step 5: Commit**

```bash
git add churn_ranker/modeling.py tests/test_modeling.py
git commit -m "feat: calibrated ChurnRanker with OOF CV and fixed tier thresholds"
```

---

### Task 6: CLI (audit / train / score) + real-sample smoke test

**Files:**
- Create: `churn_ranker/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ChurnRanker`, `RankerConfig` from Task 5; `schema`, `evaluation` from Tasks 1 and 3.
- Produces: `cli.main(argv: list[str] | None = None) -> None` with subcommands:
  - `audit <csv> [<csv> ...] --output <json>` — per-file rows/columns/snapshot dates/churn rate/usable feature count/non-numeric columns, plus pairwise MSISDN overlap counts.
  - `train <train_csv> [--eval-csv <csv>] [--artifact <path>] [--report-prefix <prefix>]` — fits, saves artifact, writes `<prefix>_training_report.json`; with `--eval-csv` also filters out MSISDNs seen in training, writes `<prefix>_validation_lift.csv`, and embeds validation metrics in the report.
  - `score <input_csv> [--artifact <path>] [--output <csv>] [--chunk-size <int>]` — chunked scoring; output columns: all `MSISDN*` columns plus `churn_probability`, `risk_tier`, `reason_code`.

- [ ] **Step 1: Write the failing CLI tests**

`tests/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_ranker.cli'`.

- [ ] **Step 3: Implement the CLI**

`churn_ranker/cli.py`:

```python
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
    report_path = f"{report_prefix}_training_report.json"
    _ensure_parent(report_path)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_cli.py -v`
Expected: 5 passed (the sample smoke test trains on 7,008 rows; allow ~1–2 min).

- [ ] **Step 5: Run the entire suite**

Run: `./venv/Scripts/python.exe -m pytest tests/test_schema.py tests/test_features.py tests/test_evaluation.py tests/test_tiers.py tests/test_modeling.py tests/test_cli.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add churn_ranker/cli.py tests/test_cli.py
git commit -m "feat: audit/train/score CLI with overlap-filtered validation and chunked scoring"
```

---

### Task 7: Real-data run — train on Feb, validate on Jan-15 snapshot, results report

**Files:**
- Modify: `.gitignore` (add outputs dir)
- Create: `docs/reports/2026-08-18-churn-ranker-results.md`
- Generated (NOT committed): `churn_ranker_outputs/audit_report.json`, `churn_ranker_outputs/churn_ranker.joblib`, `churn_ranker_outputs/churn_ranker_training_report.json`, `churn_ranker_outputs/churn_ranker_validation_lift.csv`, `churn_ranker_outputs/march_scores.csv`

**Interfaces:**
- Consumes: the CLI from Task 6.
- Produces: a written results report; no code interfaces.

- [ ] **Step 1: Gitignore the outputs directory**

Append to `.gitignore`:

```text
churn_ranker_outputs/
```

- [ ] **Step 2: Run the audit on all three files**

Run:

```bash
./venv/Scripts/python.exe -m churn_ranker.cli audit Feb1_Train_with_recharg.csv March_validation_with_recharg.csv Sample_data_full_feature.csv --output churn_ranker_outputs/audit_report.json
```

Expected (verify against `audit_report.json`): Feb ≈ 521,207 rows / churn rate ≈ 0.0407 / snapshot 2026-02-01; "March" file ≈ 431,111 rows / churn ≈ 0.0463 / snapshot **2026-01-15**; sample 7,008 rows / churn ≈ 0.0054; Feb↔Jan MSISDN overlap ≈ 40,929.

- [ ] **Step 3: Train on Feb with out-of-sample evaluation**

Run (expect 10–30 minutes on 521K rows; do not reduce the data):

```bash
./venv/Scripts/python.exe -m churn_ranker.cli train Feb1_Train_with_recharg.csv --eval-csv March_validation_with_recharg.csv --artifact churn_ranker_outputs/churn_ranker.joblib --report-prefix churn_ranker_outputs/churn_ranker
```

Expected: `churn_ranker_training_report.json` exists with non-null `oof_metrics.roc_auc`, a `validation` block with `overlap_msisdns_removed` ≈ 40,929, and `churn_ranker_validation_lift.csv` with 5 rows.

- [ ] **Step 4: Score the validation file for the deliverable list**

Run:

```bash
./venv/Scripts/python.exe -m churn_ranker.cli score March_validation_with_recharg.csv --artifact churn_ranker_outputs/churn_ranker.joblib --output churn_ranker_outputs/march_scores.csv
```

Expected: `march_scores.csv` with 431,111 rows and columns `MSISDN*`, `churn_probability`, `risk_tier`, `reason_code`; tier volumes roughly 1% / 4% / 10% of rows.

- [ ] **Step 5: Write the results report**

Create `docs/reports/2026-08-18-churn-ranker-results.md` using this template, filling every `<...>` from `churn_ranker_training_report.json` and `churn_ranker_validation_lift.csv` (copy real numbers — never estimates):

```markdown
# Churn Ranker Results — trained 2026-08-18

Spec: docs/superpowers/specs/2026-08-18-churn-risk-ranking-design.md

## Setup
- Train: Feb1_Train_with_recharg.csv (<n_rows> rows, <n_churners> churners, snapshot 2026-02-01)
- Validation: March_validation_with_recharg.csv, snapshot 2026-01-15,
  <overlap_msisdns_removed> overlapping MSISDNs removed -> <validation n> rows evaluated
- Model: HistGradientBoostingClassifier + sigmoid calibration, <n_features> features, <cv_folds>-fold OOF

## Ranking quality
| Split | ROC-AUC | PR-AUC | Base rate |
|---|---|---|---|
| OOF (train) | <...> | <...> | <...> |
| Validation | <...> | <...> | <...> |

## Validation lift table
| Top % | Contacted | Churners caught | Precision | Recall | Lift |
|---|---|---|---|---|---|
| 1% | <...> | <...> | <...> | <...> | <...> |
| 2% | <...> | <...> | <...> | <...> | <...> |
| 5% | <...> | <...> | <...> | <...> | <...> |
| 10% | <...> | <...> | <...> | <...> | <...> |
| 20% | <...> | <...> | <...> | <...> | <...> |

## Reading this
At a 4.6% base rate, random contact catches 4.6 churners per 100 calls. The lift column
is the multiplier this model achieves at each capacity. Tier sizes (1% / 5% / 15%
cumulative) can be re-fit to actual campaign capacity via RankerConfig.tier_spec.

## Caveats
- The validation snapshot (2026-01-15) precedes the training snapshot (2026-02-01);
  it is out-of-sample (disjoint customers after overlap removal) but not strictly
  out-of-time. A forward snapshot should be evaluated when available.
- If validation lift@5% is below 3.0, treat the model as not yet deployable and
  investigate before promoting (feature drift between files is the first suspect —
  compare `usable_feature_columns` across files in audit_report.json).

## Comparison with previous approaches
- unified_churn_engine (binary, precision floor 0.70): flagged 1 of 2,103 eval rows
  (recall 9%), trained on the 38-churner sample.
- churn_rules_v4: flagged 103,590 of 521,207 (19.9%) with no measured precision.
- This ranker: capacity-controlled volumes with measured precision above.
```

- [ ] **Step 6: Verify no data files are staged, then commit**

Run: `git status --short` — confirm only `.gitignore` and `docs/reports/2026-08-18-churn-ranker-results.md` are staged/untracked (no CSVs, no joblib).

```bash
git add .gitignore docs/reports/2026-08-18-churn-ranker-results.md
git commit -m "docs: churn ranker real-data results (Feb train, Jan-15 validation)"
```

---

## Self-Review (completed 2026-08-18)

- **Spec coverage:** schema harmonization → Task 1; derived intersection features → Task 2; ranking metrics/lift → Task 3; tiers + reason codes → Task 4; calibrated model with train-time thresholds → Task 5; CLI, chunked scoring, audit, overlap-filtered validation → Task 6; real-data acceptance run + documented caveats → Task 7. Non-goals honored (no new runtime deps, old scripts untouched).
- **Placeholder scan:** every code step contains complete runnable code; report template placeholders are explicitly instructions to copy real numbers from generated artifacts.
- **Type consistency:** `FE_*` names in Tasks 2/4/5 match (`FE_DATA_W13_RATIO`, `FE_OG_VOICE_W13_RATIO`, `FE_ALL_CORE_ZERO_W13`, `FE_TERMINAL_MULTI_SERVICE`, `FE_RECHARGE_STOPPED`, `FE_DECLINING_SERVICES`); `tiers.tier_thresholds/assign_tiers/reason_codes` signatures match their call sites in `modeling.py`; `evaluation.ranking_metrics/lift_table` match `cli.py` usage; fixture `synthetic` defined in Task 1, consumed in Tasks 5–6.
