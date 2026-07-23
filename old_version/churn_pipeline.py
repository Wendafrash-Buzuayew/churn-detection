"""
churn_pipeline.py
=================
Production-grade Telecom Churn Prediction Pipeline.
Handles extreme class imbalance (~0.5% churn rate) via:
  - Dynamic scale_pos_weight
  - StratifiedKFold (5-fold) cross-validation
  - F2-score threshold sweep (0.01 → 0.50)
  - Heavy XGBoost regularisation (max_depth=3, reg_lambda=15, reg_alpha=2)

Supports two data modes:
  - ORACLE  : oracledb connection to CVM_DM_PROD table
  - CSV     : flat file with at minimum weekly W10-W13 raw metrics

Run:  python churn_pipeline.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS & PACKAGE GUARDS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
import logging
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – swap to "TkAgg" if you want a live window
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    _XGB_OK = True
except ImportError:
    _XGB_OK = False
    log.error("XGBoost not installed. Run: pip install xgboost")

try:
    import oracledb
    _ORA_OK = True
except ImportError:
    _ORA_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG: Dict = {
    # ── Data source ──────────────────────────────────────────────────────────
    "RUN_MODE"        : os.getenv("RUN_MODE", "CSV"),        # "ORACLE" | "CSV"
    "INPUT_CSV"       : os.getenv("INPUT_CSV", "Sample_data.csv"),

    # ── Oracle credentials (override via env vars in production) ─────────────
    "ORA_HOST"        : os.getenv("ORA_HOST",    "mdc1-charli-scan.safaricomet.net"),
    "ORA_PORT"        : int(os.getenv("ORA_PORT", "1521")),
    "ORA_SERVICE"     : os.getenv("ORA_SERVICE",  "DMCVLIVE.safaricomet.net"),
    "ORA_USER"        : os.getenv("ORA_USER",     "CVM_DM_PROD"),
    "ORA_PASSWORD"    : os.getenv("ORA_PASSWORD", ""),        # always inject via env
    "ORA_TABLE"       : os.getenv("ORA_TABLE",    "CVM_DM_PROD.CHURN_POC_JAN15_FULL_FEATURES_V2"),

    # ── Modelling ─────────────────────────────────────────────────────────────
    "TARGET"          : "LABEL_CHURN_90D",
    "RANDOM_STATE"    : 42,
    "N_FOLDS"         : 5,
    "THRESHOLD_MIN"   : 0.01,
    "THRESHOLD_MAX"   : 0.50,
    "THRESHOLD_STEPS" : 100,

    # ── XGBoost hyperparameters (heavy regularisation for only 50 positives) ─
    "XGB_PARAMS": {
        "n_estimators"    : 600,
        "max_depth"       : 3,           # RULE 4
        "learning_rate"   : 0.02,        # RULE 4
        "subsample"       : 0.80,
        "colsample_bytree": 0.80,
        "min_child_weight": 10,
        "reg_lambda"      : 15.0,        # RULE 4
        "reg_alpha"       : 2.0,         # RULE 4
        "tree_method"     : "hist",      # RULE 4
        "objective"       : "binary:logistic",
        "eval_metric"     : "aucpr",
        "n_jobs"          : -1,
    },

    # ── Output ────────────────────────────────────────────────────────────────
    "OUTPUT_DIR"      : "./churn_pipeline_outputs",
}

os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPLICIT 79-FEATURE CONTRACT
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FEATURES: List[str] = [
    # ── Data Usage Behavior (9) ───────────────────────────────────────────────
    "DATA_MB_TREND_SLOPE_13W",
    "DATA_MB_VOLATILITY_13W",
    "DATA_MB_CONSEC_ZERO_WEEKS_13W",
    "DATA_MB_RECENT_SHARE_13W",
    "DATA_MB_RECENT_VS_PREV_DROP_PCT",
    "DATA_MB_PEAK_TO_RECENT_DROP_PCT",
    "DATA_MB_W13_VS_W12_DROP_PCT",
    "ZERO_DATA_W13_FLAG",
    "ZERO_DATA_RECENT_4W_FLAG",

    # ── Financial Revenue (9) ─────────────────────────────────────────────────
    "TOTAL_REVENUE_TREND_SLOPE_13W",
    "TOTAL_REVENUE_VOLATILITY_13W",
    "TOTAL_REVENUE_CONSEC_ZERO_WEEKS_13W",
    "TOTAL_REVENUE_RECENT_SHARE_13W",
    "TOTAL_REVENUE_RECENT_VS_PREV_DROP_PCT",
    "TOTAL_REVENUE_PEAK_TO_RECENT_DROP_PCT",
    "TOTAL_REVENUE_W13_VS_W12_DROP_PCT",
    "ZERO_REVENUE_W13_FLAG",
    "ZERO_REVENUE_RECENT_4W_FLAG",

    # ── Package / Bundle Engagement (9) ──────────────────────────────────────
    "BUNDLE_CNT_TREND_SLOPE_13W",
    "BUNDLE_CNT_VOLATILITY_13W",
    "BUNDLE_CNT_CONSEC_ZERO_WEEKS_13W",
    "BUNDLE_CNT_RECENT_SHARE_13W",
    "BUNDLE_CNT_RECENT_VS_PREV_DROP_PCT",
    "BUNDLE_CNT_PEAK_TO_RECENT_DROP_PCT",
    "BUNDLE_CNT_W13_VS_W12_DROP_PCT",
    "ZERO_BUNDLE_W13_FLAG",
    "ZERO_BUNDLE_RECENT_4W_FLAG",

    # ── Total Voice Minutes (9) ───────────────────────────────────────────────
    "TOTAL_VOICE_MIN_TREND_SLOPE_13W",
    "TOTAL_VOICE_MIN_VOLATILITY_13W",
    "TOTAL_VOICE_MIN_CONSEC_ZERO_WEEKS_13W",
    "TOTAL_VOICE_MIN_RECENT_SHARE_13W",
    "TOTAL_VOICE_MIN_RECENT_VS_PREV_DROP_PCT",
    "TOTAL_VOICE_MIN_PEAK_TO_RECENT_DROP_PCT",
    "TOTAL_VOICE_MIN_W13_VS_W12_DROP_PCT",
    "ZERO_VOICE_W13_FLAG",
    "ZERO_VOICE_RECENT_4W_FLAG",

    # ── Outgoing Voice (7) ────────────────────────────────────────────────────
    "OG_VOICE_MIN_TREND_SLOPE_13W",
    "OG_VOICE_MIN_VOLATILITY_13W",
    "OG_VOICE_MIN_CONSEC_ZERO_WEEKS_13W",
    "OG_VOICE_MIN_RECENT_SHARE_13W",
    "OG_VOICE_MIN_RECENT_VS_PREV_DROP_PCT",
    "OG_VOICE_MIN_PEAK_TO_RECENT_DROP_PCT",
    "OG_VOICE_MIN_W13_VS_W12_DROP_PCT",

    # ── Incoming Voice (7) ────────────────────────────────────────────────────
    "IC_VOICE_MIN_TREND_SLOPE_13W",
    "IC_VOICE_MIN_VOLATILITY_13W",
    "IC_VOICE_MIN_CONSEC_ZERO_WEEKS_13W",
    "IC_VOICE_MIN_RECENT_SHARE_13W",
    "IC_VOICE_MIN_RECENT_VS_PREV_DROP_PCT",
    "IC_VOICE_MIN_PEAK_TO_RECENT_DROP_PCT",
    "IC_VOICE_MIN_W13_VS_W12_DROP_PCT",

    # ── Total SMS (9) ─────────────────────────────────────────────────────────
    "TOTAL_SMS_COUNT_TREND_SLOPE_13W",
    "TOTAL_SMS_COUNT_VOLATILITY_13W",
    "TOTAL_SMS_COUNT_CONSEC_ZERO_WEEKS_13W",
    "TOTAL_SMS_COUNT_RECENT_SHARE_13W",
    "TOTAL_SMS_COUNT_RECENT_VS_PREV_DROP_PCT",
    "TOTAL_SMS_COUNT_PEAK_TO_RECENT_DROP_PCT",
    "TOTAL_SMS_COUNT_W13_VS_W12_DROP_PCT",
    "ZERO_SMS_W13_FLAG",
    "ZERO_SMS_RECENT_4W_FLAG",

    # ── Outgoing SMS (7) ──────────────────────────────────────────────────────
    "OG_SMS_COUNT_TREND_SLOPE_13W",
    "OG_SMS_COUNT_VOLATILITY_13W",
    "OG_SMS_COUNT_CONSEC_ZERO_WEEKS_13W",
    "OG_SMS_COUNT_RECENT_SHARE_13W",
    "OG_SMS_COUNT_RECENT_VS_PREV_DROP_PCT",
    "OG_SMS_COUNT_PEAK_TO_RECENT_DROP_PCT",
    "OG_SMS_COUNT_W13_VS_W12_DROP_PCT",

    # ── Incoming SMS (7) ──────────────────────────────────────────────────────
    "IC_SMS_COUNT_TREND_SLOPE_13W",
    "IC_SMS_COUNT_VOLATILITY_13W",
    "IC_SMS_COUNT_CONSEC_ZERO_WEEKS_13W",
    "IC_SMS_COUNT_RECENT_SHARE_13W",
    "IC_SMS_COUNT_RECENT_VS_PREV_DROP_PCT",
    "IC_SMS_COUNT_PEAK_TO_RECENT_DROP_PCT",
    "IC_SMS_COUNT_W13_VS_W12_DROP_PCT",

    # ── Multi-window Activity & Cross-sell Diversity Ecosystem (9) ───────────
    "ANY_ACTIVE_WEEKS_13W",
    "ANY_ACTIVE_WEEKS_RECENT_4W",
    "ANY_ACTIVE_WEEKS_PREV_4W",
    "ANY_ACTIVE_WEEKS_OLDER_5W",
    "SERVICE_DIVERSITY_RECENT_4W",
    "SERVICE_DIVERSITY_PREV_4W",
    "SERVICE_DIVERSITY_DROP_FLAG",
    "SERVICE_DIVERSITY_DROP_PCT",
    "ENGAGEMENT_DROP_SCORE",
]

assert len(MODEL_FEATURES) == 82, f"Expected 82 features, got {len(MODEL_FEATURES)}"
log.info("Feature contract loaded: %d features", len(MODEL_FEATURES))


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA CONNECTORS
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_oracle() -> pd.DataFrame:
    """Pull full feature table from Oracle.  Requires oracledb and credentials."""
    if not _ORA_OK:
        raise ImportError("oracledb not installed. pip install oracledb")
    dsn = oracledb.makedsn(
        host=CONFIG["ORA_HOST"],
        port=CONFIG["ORA_PORT"],
        service_name=CONFIG["ORA_SERVICE"],
    )
    conn = oracledb.connect(
        user=CONFIG["ORA_USER"],
        password=CONFIG["ORA_PASSWORD"],
        dsn=dsn,
    )
    sql = f"SELECT * FROM {CONFIG['ORA_TABLE']}"
    log.info("Querying Oracle: %s", CONFIG["ORA_TABLE"])
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def _load_from_csv() -> pd.DataFrame:
    csv_path = CONFIG["INPUT_CSV"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    log.info("Loading CSV: %s", csv_path)
    return pd.read_csv(csv_path)


def load_raw_data() -> pd.DataFrame:
    """Dispatcher: Oracle or CSV based on RUN_MODE."""
    mode = CONFIG["RUN_MODE"].upper()
    if mode == "ORACLE":
        df = _load_from_oracle()
    else:
        df = _load_from_csv()
    df.columns = [c.upper().strip() for c in df.columns]
    log.info("Raw data loaded: %s rows × %s cols", *df.shape)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

# Mapping: (feature_prefix, list_of_weekly_column_suffixes)
# The pipeline auto-detects available weeks, so it works whether the
# source table carries W1-W13 (Oracle) or only W10-W13 (sample CSV).
_METRIC_WEEKLY_MAP = {
    "DATA_MB"         : ("DATA_MB",          "DATA"),
    "TOTAL_REVENUE"   : ("TOTAL_REVENUE",     "REVENUE"),   # synthesised if absent
    "BUNDLE_CNT"      : ("BUNDLE_CNT",        "BUNDLE"),
    "TOTAL_VOICE_MIN" : ("TOTAL_VOICE_MIN",   "VOICE"),
    "OG_VOICE_MIN"    : ("OG_VOICE_MIN",      None),
    "IC_VOICE_MIN"    : ("IC_VOICE_MIN",      None),
    "TOTAL_SMS_COUNT" : ("TOTAL_SMS_COUNT",   "SMS"),
    "OG_SMS_COUNT"    : ("OG_SMS_COUNT",      None),
    "IC_SMS_COUNT"    : ("IC_SMS_COUNT",      None),
}

_ALL_POSSIBLE_WEEKS = [f"W{i}" for i in range(1, 14)]   # W1 … W13


def _available_weekly_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    """Return the sorted weekly columns that actually exist in df."""
    cols = [f"{prefix}_{w}" for w in _ALL_POSSIBLE_WEEKS if f"{prefix}_{w}" in df.columns]
    return cols


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _ols_slope(vals: np.ndarray) -> np.ndarray:
    """
    Vectorised OLS slope for each row.
    vals: (n_rows, n_weeks) float array – already cleaned of NaN/Inf.
    """
    n = vals.shape[1]
    x = np.arange(n, dtype=float)
    x_c = x - x.mean()
    denom = (x_c ** 2).sum()
    if denom == 0:
        return np.zeros(vals.shape[0])
    y_c = vals - vals.mean(axis=1, keepdims=True)
    return (y_c * x_c).sum(axis=1) / denom


def _consec_zero_from_end(vals: np.ndarray) -> np.ndarray:
    """Count of consecutive zero weeks counting backwards from the last column."""
    zero = vals <= 0
    out = np.zeros(zero.shape[0], dtype=int)
    n = zero.shape[1]
    for j in range(n - 1, -1, -1):
        still_zero = zero[:, j] & (out == (n - 1 - j))
        out += still_zero.astype(int)
    # reset counter where a non-zero week was encountered before reaching column j
    # recompute properly with a simple loop (fast for small n ≤ 13)
    out2 = np.zeros(zero.shape[0], dtype=int)
    for i in range(zero.shape[0]):
        c = 0
        for j in range(n - 1, -1, -1):
            if zero[i, j]:
                c += 1
            else:
                break
        out2[i] = c
    return out2


def _bounded_drop_pct(prior: np.ndarray, recent: np.ndarray) -> np.ndarray:
    """
    (prior − recent) / max(|prior|, |recent|, 1)
    Bounded to [−1, +1], prevents explosion when both values are near zero.
    """
    denom = np.maximum(np.maximum(np.abs(prior), np.abs(recent)), 1.0)
    return np.clip((prior - recent) / denom, -1.0, 1.0)


def _synthesise_revenue_weekly(df: pd.DataFrame, weekly_cols: List[str]) -> pd.DataFrame:
    """
    When TOTAL_REVENUE_Wxx are absent (CSV mode), synthesise per-week values
    by distributing each revenue component proportionally to weekly usage volume.

    Components and their weekly usage proxies:
      DATA_REVENUE     → DATA_MB_Wxx
      VOICE_REVENUE    → TOTAL_VOICE_MIN_Wxx
      SMS_REVENUE      → TOTAL_SMS_COUNT_Wxx
      BUNDLE_REVENUE   → BUNDLE_CNT_Wxx
    """
    df = df.copy()
    week_suffixes = [c.replace("TOTAL_REVENUE_", "") for c in weekly_cols]

    component_map = {
        "DATA_REVENUE_RECENT_4W"   : "DATA_MB",
        "VOICE_REVENUE_RECENT_4W"  : "TOTAL_VOICE_MIN",
        "SMS_REVENUE_RECENT_4W"    : "TOTAL_SMS_COUNT",
        "BUNDLE_REVENUE_RECENT_4W" : "BUNDLE_CNT",
    }

    for dest_col in weekly_cols:
        df[dest_col] = 0.0

    for rev_col, usage_prefix in component_map.items():
        if rev_col not in df.columns:
            continue
        rev_4w = _safe_numeric(df[rev_col])
        usage_wks = []
        for ws in week_suffixes:
            uc = f"{usage_prefix}_{ws}"
            usage_wks.append(_safe_numeric(df[uc]) if uc in df.columns else pd.Series(0.0, index=df.index))
        usage_total = sum(usage_wks) + 1.0   # +1 prevents div/0

        for ws, usage_w in zip(week_suffixes, usage_wks):
            df[f"TOTAL_REVENUE_{ws}"] += rev_4w * (usage_w / usage_total)

    return df


def _compute_weekly_dynamics(
    df: pd.DataFrame,
    feature_prefix: str,
    zero_flag_token: Optional[str],
    zero_recent_4w_col: Optional[str],
) -> pd.DataFrame:
    """
    Compute the 7 (or 9 with zero flags) dynamic features for one metric.

    Parameters
    ----------
    feature_prefix      e.g. "DATA_MB"
    zero_flag_token     e.g. "DATA"  → column name ZERO_DATA_W13_FLAG
    zero_recent_4w_col  e.g. "ZERO_DATA_RECENT_4W_FLAG" (pass-through if exists)
    """
    weekly_cols = _available_weekly_cols(df, feature_prefix)

    # ── Fallback: synthesise TOTAL_REVENUE weekly if needed ──────────────────
    if feature_prefix == "TOTAL_REVENUE" and not weekly_cols:
        # Request at least W10-W13 as synthetic destination columns
        desired = [f"TOTAL_REVENUE_{w}" for w in ["W10", "W11", "W12", "W13"]]
        df = _synthesise_revenue_weekly(df, desired)
        weekly_cols = _available_weekly_cols(df, feature_prefix)

    if len(weekly_cols) < 2:
        log.warning("Not enough weekly columns for %s (%d found). Zeroing features.", feature_prefix, len(weekly_cols))
        zero_out = {}
        for feat in MODEL_FEATURES:
            if feat.startswith(feature_prefix) or (zero_flag_token and f"ZERO_{zero_flag_token}_" in feat):
                zero_out[feat] = 0.0
        return pd.DataFrame(zero_out, index=df.index)

    vals = df[weekly_cols].apply(pd.to_numeric, errors="coerce") \
                          .replace([np.inf, -np.inf], np.nan) \
                          .fillna(0.0) \
                          .values.astype(float)

    n_weeks = vals.shape[1]
    last     = vals[:, -1]
    prev_4   = vals[:, max(0, n_weeks - 5):n_weeks - 1]   # up to 4 weeks prior to last
    prior_mean = prev_4.mean(axis=1) if prev_4.size else np.zeros(len(df))
    second_last = vals[:, -2] if n_weeks >= 2 else np.zeros(len(df))
    window_max  = vals.max(axis=1)
    window_sum  = vals.sum(axis=1)

    out = {}
    out[f"{feature_prefix}_TREND_SLOPE_13W"]          = _ols_slope(vals)
    out[f"{feature_prefix}_VOLATILITY_13W"]            = vals.std(axis=1, ddof=0)
    out[f"{feature_prefix}_CONSEC_ZERO_WEEKS_13W"]     = _consec_zero_from_end(vals)
    out[f"{feature_prefix}_RECENT_SHARE_13W"]          = last / (np.abs(window_sum) + 1.0)
    out[f"{feature_prefix}_RECENT_VS_PREV_DROP_PCT"]   = _bounded_drop_pct(prior_mean, last)
    out[f"{feature_prefix}_PEAK_TO_RECENT_DROP_PCT"]   = _bounded_drop_pct(window_max, last)
    out[f"{feature_prefix}_W13_VS_W12_DROP_PCT"]       = _bounded_drop_pct(second_last, last)

    if zero_flag_token:
        out[f"ZERO_{zero_flag_token}_W13_FLAG"] = (last <= 0).astype(int)

        if zero_recent_4w_col and zero_recent_4w_col in df.columns:
            out[f"ZERO_{zero_flag_token}_RECENT_4W_FLAG"] = (
                _safe_numeric(df[zero_recent_4w_col]).astype(int).values
            )
        else:
            out[f"ZERO_{zero_flag_token}_RECENT_4W_FLAG"] = (window_sum <= 0).astype(int)

    return pd.DataFrame(out, index=df.index)


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build all 82 behavioural features from the raw weekly columns.
    Works with both the full 13-week Oracle table and the sample 4-week CSV.
    Returns the original dataframe augmented with all feature columns.
    """
    df = raw.copy()

    # Normalise all numeric-intended columns
    id_cols = {"MSISDN", "MSISDN_9", "MSISDN_251", "SNAPSHOT_DATE",
               "LABEL_CHURN_90D", "DATASET_TYPE"}
    for col in df.columns:
        if col not in id_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce") \
                        .replace([np.inf, -np.inf], np.nan) \
                        .fillna(0.0)

    # ── Per-metric dynamics ───────────────────────────────────────────────────
    dynamics_spec = [
        # (feature_prefix,       zero_token,  zero_recent_4w_source_col)
        ("DATA_MB",          "DATA",     "ZERO_DATA_RECENT_4W_FLAG"),
        ("TOTAL_REVENUE",    "REVENUE",  None),
        ("BUNDLE_CNT",       "BUNDLE",   "ZERO_BUNDLE_RECENT_4W_FLAG"),
        ("TOTAL_VOICE_MIN",  "VOICE",    "ZERO_VOICE_RECENT_4W_FLAG"),
        ("OG_VOICE_MIN",     None,       None),
        ("IC_VOICE_MIN",     None,       None),
        ("TOTAL_SMS_COUNT",  "SMS",      "ZERO_SMS_RECENT_4W_FLAG"),
        ("OG_SMS_COUNT",     None,       None),
        ("IC_SMS_COUNT",     None,       None),
    ]

    dyn_frames = []
    for prefix, zero_tok, zero_src in dynamics_spec:
        block = _compute_weekly_dynamics(df, prefix, zero_tok, zero_src)
        dyn_frames.append(block)

    dyn_df = pd.concat(dyn_frames, axis=1)

    # ── Revenue zero flags from TOTAL_REVENUE zero columns ───────────────────
    # The zero-flag token for revenue is "REVENUE" → ZERO_REVENUE_W13_FLAG
    # This is already handled inside _compute_weekly_dynamics for TOTAL_REVENUE.

    # ── Ecosystem / Activity features ────────────────────────────────────────
    eco = {}

    # ANY_ACTIVE_WEEKS_* : a subscriber-week is active if any metric > 0
    activity_prefixes = ["DATA_MB", "TOTAL_VOICE_MIN", "TOTAL_SMS_COUNT", "BUNDLE_CNT"]

    # Build per-week activity flag across all available weeks
    all_weeks_union = sorted(
        set(
            w for p in activity_prefixes
            for w in _ALL_POSSIBLE_WEEKS
            if f"{p}_{w}" in df.columns
        ),
        key=lambda w: int(w[1:])
    )

    def _active_in_week(w: str) -> pd.Series:
        cols = [f"{p}_{w}" for p in activity_prefixes if f"{p}_{w}" in df.columns]
        if not cols:
            return pd.Series(0, index=df.index)
        return (df[cols].sum(axis=1) > 0).astype(int)

    per_week_active = {w: _active_in_week(w) for w in all_weeks_union}

    # ANY_ACTIVE_WEEKS_13W : total active weeks across full history
    eco["ANY_ACTIVE_WEEKS_13W"] = pd.concat(per_week_active.values(), axis=1).sum(axis=1)

    # Recent 4W = last 4 available weeks
    recent_4w = all_weeks_union[-4:] if len(all_weeks_union) >= 4 else all_weeks_union
    prev_4w   = all_weeks_union[-8:-4] if len(all_weeks_union) >= 8 else all_weeks_union[:max(1, len(all_weeks_union) - 4)]
    older_5w  = all_weeks_union[:-4]  if len(all_weeks_union) > 4  else []

    eco["ANY_ACTIVE_WEEKS_RECENT_4W"] = (
        pd.concat([per_week_active[w] for w in recent_4w], axis=1).sum(axis=1)
        if recent_4w else pd.Series(0, index=df.index)
    )
    eco["ANY_ACTIVE_WEEKS_PREV_4W"] = (
        pd.concat([per_week_active[w] for w in prev_4w], axis=1).sum(axis=1)
        if prev_4w else pd.Series(0, index=df.index)
    )
    eco["ANY_ACTIVE_WEEKS_OLDER_5W"] = (
        pd.concat([per_week_active[w] for w in older_5w], axis=1).sum(axis=1)
        if older_5w else pd.Series(0, index=df.index)
    )

    # SERVICE_DIVERSITY_RECENT_4W : prefer source column, else recompute
    if "SERVICE_DIVERSITY_RECENT_4W" in df.columns:
        eco["SERVICE_DIVERSITY_RECENT_4W"] = _safe_numeric(df["SERVICE_DIVERSITY_RECENT_4W"])
    else:
        svc_flags_r = [
            (df[[f"DATA_MB_{w}" for w in recent_4w if f"DATA_MB_{w}" in df.columns]].sum(axis=1) > 0).astype(int)
            if any(f"DATA_MB_{w}" in df.columns for w in recent_4w) else pd.Series(0, index=df.index),
            (df[[f"TOTAL_VOICE_MIN_{w}" for w in recent_4w if f"TOTAL_VOICE_MIN_{w}" in df.columns]].sum(axis=1) > 0).astype(int)
            if any(f"TOTAL_VOICE_MIN_{w}" in df.columns for w in recent_4w) else pd.Series(0, index=df.index),
            (df[[f"TOTAL_SMS_COUNT_{w}" for w in recent_4w if f"TOTAL_SMS_COUNT_{w}" in df.columns]].sum(axis=1) > 0).astype(int)
            if any(f"TOTAL_SMS_COUNT_{w}" in df.columns for w in recent_4w) else pd.Series(0, index=df.index),
            (df[[f"BUNDLE_CNT_{w}" for w in recent_4w if f"BUNDLE_CNT_{w}" in df.columns]].sum(axis=1) > 0).astype(int)
            if any(f"BUNDLE_CNT_{w}" in df.columns for w in recent_4w) else pd.Series(0, index=df.index),
        ]
        eco["SERVICE_DIVERSITY_RECENT_4W"] = pd.concat(svc_flags_r, axis=1).sum(axis=1)

    # SERVICE_DIVERSITY_PREV_4W
    if prev_4w:
        svc_flags_p = [
            (df[[f"DATA_MB_{w}" for w in prev_4w if f"DATA_MB_{w}" in df.columns]].sum(axis=1) > 0).astype(int)
            if any(f"DATA_MB_{w}" in df.columns for w in prev_4w) else pd.Series(0, index=df.index),
            (df[[f"TOTAL_VOICE_MIN_{w}" for w in prev_4w if f"TOTAL_VOICE_MIN_{w}" in df.columns]].sum(axis=1) > 0).astype(int)
            if any(f"TOTAL_VOICE_MIN_{w}" in df.columns for w in prev_4w) else pd.Series(0, index=df.index),
            (df[[f"TOTAL_SMS_COUNT_{w}" for w in prev_4w if f"TOTAL_SMS_COUNT_{w}" in df.columns]].sum(axis=1) > 0).astype(int)
            if any(f"TOTAL_SMS_COUNT_{w}" in df.columns for w in prev_4w) else pd.Series(0, index=df.index),
            (df[[f"BUNDLE_CNT_{w}" for w in prev_4w if f"BUNDLE_CNT_{w}" in df.columns]].sum(axis=1) > 0).astype(int)
            if any(f"BUNDLE_CNT_{w}" in df.columns for w in prev_4w) else pd.Series(0, index=df.index),
        ]
        eco["SERVICE_DIVERSITY_PREV_4W"] = pd.concat(svc_flags_p, axis=1).sum(axis=1)
    else:
        eco["SERVICE_DIVERSITY_PREV_4W"] = eco["SERVICE_DIVERSITY_RECENT_4W"].copy()

    eco_df = pd.DataFrame(eco, index=df.index)
    eco_df["SERVICE_DIVERSITY_DROP_FLAG"] = (
        eco_df["SERVICE_DIVERSITY_PREV_4W"] > eco_df["SERVICE_DIVERSITY_RECENT_4W"]
    ).astype(int)
    eco_df["SERVICE_DIVERSITY_DROP_PCT"] = _bounded_drop_pct(
        eco_df["SERVICE_DIVERSITY_PREV_4W"].values,
        eco_df["SERVICE_DIVERSITY_RECENT_4W"].values,
    )

    # ENGAGEMENT_DROP_SCORE : mean of the 4 most sensitive drop signals, clipped [0,1]
    drop_signals = [
        "DATA_MB_RECENT_VS_PREV_DROP_PCT",
        "TOTAL_VOICE_MIN_RECENT_VS_PREV_DROP_PCT",
        "TOTAL_SMS_COUNT_RECENT_VS_PREV_DROP_PCT",
        "BUNDLE_CNT_RECENT_VS_PREV_DROP_PCT",
        "TOTAL_REVENUE_RECENT_VS_PREV_DROP_PCT",
        "SERVICE_DIVERSITY_DROP_PCT",
    ]
    drop_avail = [c for c in drop_signals if c in dyn_df.columns] + \
                 [c for c in drop_signals if c in eco_df.columns]
    drop_avail = list(dict.fromkeys(drop_avail))   # deduplicate, preserve order

    if drop_avail:
        parts = []
        for c in drop_avail:
            parts.append(dyn_df[c].values if c in dyn_df.columns else eco_df[c].values)
        eco_df["ENGAGEMENT_DROP_SCORE"] = np.clip(np.stack(parts, axis=1).mean(axis=1), -1.0, 1.0)
    else:
        eco_df["ENGAGEMENT_DROP_SCORE"] = 0.0

    # ── Assemble the final feature matrix ────────────────────────────────────
    feature_df = pd.concat([dyn_df, eco_df], axis=1)

    # Validate that every declared model feature is present
    missing_feats = [f for f in MODEL_FEATURES if f not in feature_df.columns]
    if missing_feats:
        log.warning("Filling %d missing model features with 0: %s", len(missing_feats), missing_feats)
        for mf in missing_feats:
            feature_df[mf] = 0.0

    # Final safety: replace any residual inf / NaN
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    log.info("Feature engineering complete. Feature matrix: %s rows × %s cols", *feature_df.shape)
    return pd.concat([raw[[CONFIG["TARGET"]]], feature_df[MODEL_FEATURES]], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PREPROCESSING  (cast + scale)
# ─────────────────────────────────────────────────────────────────────────────

# Features that are already bounded integers / flags – do NOT scale these
_FLAG_FEATURES = {f for f in MODEL_FEATURES if "FLAG" in f or "CONSEC" in f
                  or "WEEKS" in f or "DIVERSITY" in f}

def preprocess(
    feature_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str], StandardScaler]:
    """
    Returns (X, y, feature_names, fitted_scaler).
    Continuous features are StandardScaler-normalised; flag / count features kept as-is.
    """
    y = feature_df[CONFIG["TARGET"]].astype(int).values
    X_raw = feature_df[MODEL_FEATURES].copy()

    continuous_cols = [c for c in MODEL_FEATURES if c not in _FLAG_FEATURES]
    flag_cols       = [c for c in MODEL_FEATURES if c in _FLAG_FEATURES]

    scaler = StandardScaler()
    X_cont  = scaler.fit_transform(X_raw[continuous_cols].values)
    X_flags = X_raw[flag_cols].values

    # Reconstruct in original column order
    col_order = {c: i for i, c in enumerate(continuous_cols + flag_cols)}
    X_full = np.empty((len(X_raw), len(MODEL_FEATURES)), dtype=float)
    for idx, feat in enumerate(MODEL_FEATURES):
        if feat in col_order:
            src_idx = col_order[feat]
            if feat in _FLAG_FEATURES:
                X_full[:, idx] = X_flags[:, flag_cols.index(feat)]
            else:
                X_full[:, idx] = X_cont[:, continuous_cols.index(feat)]

    log.info("Preprocessing done. X: %s, y distribution: %s", X_full.shape,
             dict(zip(*np.unique(y, return_counts=True))))
    return X_full, y, MODEL_FEATURES, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 6. STRATIFIED 5-FOLD CV with XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def run_stratified_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    5-Fold StratifiedKFold cross-validation.

    Returns
    -------
    oof_prob       : out-of-fold predicted probabilities (n_samples,)
    oof_true       : corresponding ground-truth labels  (n_samples,)
    feat_importances: mean feature importances across folds (n_features,)
    """
    if not _XGB_OK:
        raise RuntimeError("XGBoost is required. pip install xgboost")

    skf = StratifiedKFold(
        n_splits=CONFIG["N_FOLDS"],
        shuffle=True,
        random_state=CONFIG["RANDOM_STATE"],
    )

    # RULE 3: dynamic scale_pos_weight from the FULL dataset label distribution
    neg_count = int((y == 0).sum())
    pos_count = int((y == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)

    log.info("Class distribution → negative: %d | positive: %d", neg_count, pos_count)
    log.info("scale_pos_weight (dynamic) = %.4f", scale_pos_weight)

    oof_prob = np.zeros(len(y), dtype=float)
    oof_true = np.zeros(len(y), dtype=int)
    fold_importances = np.zeros(len(feature_names), dtype=float)

    xgb_params = {**CONFIG["XGB_PARAMS"],
                  "scale_pos_weight": scale_pos_weight,
                  "random_state"    : CONFIG["RANDOM_STATE"]}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        fold_neg = int((y_tr == 0).sum())
        fold_pos = int((y_tr == 1).sum())
        log.info("  Fold %d/%d | train: %d (%d pos) | val: %d (%d pos)",
                 fold_idx, CONFIG["N_FOLDS"],
                 len(y_tr), fold_pos, len(y_val), int((y_val == 1).sum()))

        model = XGBClassifier(**xgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        oof_prob[val_idx] = model.predict_proba(X_val)[:, 1]
        oof_true[val_idx] = y_val
        fold_importances  += model.feature_importances_

    fold_importances /= CONFIG["N_FOLDS"]    # average across folds

    oof_auc   = roc_auc_score(oof_true, oof_prob)
    oof_prauc = average_precision_score(oof_true, oof_prob)
    log.info("OOF ROC-AUC = %.4f | OOF PR-AUC = %.4f", oof_auc, oof_prauc)

    return oof_prob, oof_true, fold_importances


# ─────────────────────────────────────────────────────────────────────────────
# 7. THRESHOLD OPTIMISER  (maximise F2-score)
# ─────────────────────────────────────────────────────────────────────────────

def optimise_threshold(
    oof_true: np.ndarray,
    oof_prob: np.ndarray,
    th_min: float = 0.01,
    th_max: float = 0.50,
    steps : int   = 100,
) -> Tuple[float, pd.DataFrame]:
    """
    Sweep thresholds and return the one that maximises F2-score.
    F2 weights Recall twice as heavily as Precision (beta=2), minimising
    False Negatives (missed churners) at the cost of more False Positives.

    Returns
    -------
    best_threshold : float
    sweep_df       : full sweep results DataFrame
    """
    thresholds = np.linspace(th_min, th_max, steps)
    rows = []
    for th in thresholds:
        pred = (oof_prob >= th).astype(int)
        prec   = precision_score(oof_true, pred, zero_division=0)
        rec    = recall_score   (oof_true, pred, zero_division=0)
        f1     = f1_score       (oof_true, pred, zero_division=0)
        f2     = fbeta_score    (oof_true, pred, beta=2, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(oof_true, pred).ravel()
        rows.append({
            "threshold"    : round(float(th), 4),
            "precision"    : round(prec,  4),
            "recall"       : round(rec,   4),
            "f1"           : round(f1,    4),
            "f2"           : round(f2,    4),
            "tp"           : int(tp),
            "fp"           : int(fp),
            "fn"           : int(fn),
            "tn"           : int(tn),
            "predicted_pos": int(tp + fp),
        })

    sweep_df = pd.DataFrame(rows)
    best_row  = sweep_df.loc[sweep_df["f2"].idxmax()]
    best_th   = float(best_row["threshold"])

    log.info("Optimal threshold (max F2) = %.4f → F2=%.4f | Recall=%.4f | Precision=%.4f",
             best_th, best_row["f2"], best_row["recall"], best_row["precision"])
    return best_th, sweep_df


# ─────────────────────────────────────────────────────────────────────────────
# 8. REPORTING & VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _print_separator(char: str = "═", width: int = 72) -> None:
    print(char * width)


def report_metrics(
    oof_true     : np.ndarray,
    oof_prob     : np.ndarray,
    best_threshold: float,
    sweep_df     : pd.DataFrame,
    feature_names: List[str],
    fold_importances: np.ndarray,
    output_dir   : str,
) -> None:
    """Prints OOF metrics, confusion matrix, classification report, and saves plots."""

    oof_pred   = (oof_prob >= best_threshold).astype(int)
    oof_auc    = roc_auc_score(oof_true, oof_prob)
    oof_prauc  = average_precision_score(oof_true, oof_prob)
    oof_f2     = fbeta_score(oof_true, oof_pred, beta=2, zero_division=0)
    oof_recall = recall_score(oof_true, oof_pred, zero_division=0)
    oof_prec   = precision_score(oof_true, oof_pred, zero_division=0)

    _print_separator()
    print("  TELECOM CHURN MODEL — OUT-OF-FOLD EVALUATION SUMMARY")
    _print_separator()
    print(f"  Validation strategy : {CONFIG['N_FOLDS']}-Fold Stratified CV")
    print(f"  XGB max_depth       : {CONFIG['XGB_PARAMS']['max_depth']}   "
          f"reg_lambda={CONFIG['XGB_PARAMS']['reg_lambda']}  "
          f"reg_alpha={CONFIG['XGB_PARAMS']['reg_alpha']}  "
          f"learning_rate={CONFIG['XGB_PARAMS']['learning_rate']}")
    print(f"  Threshold strategy  : F2-maximising sweep "
          f"[{CONFIG['THRESHOLD_MIN']:.2f} → {CONFIG['THRESHOLD_MAX']:.2f}]")
    print(f"  Optimal threshold   : {best_threshold:.4f}")
    _print_separator()
    print(f"  ROC-AUC             : {oof_auc:.4f}")
    print(f"  PR-AUC              : {oof_prauc:.4f}")
    print(f"  F2-Score            : {oof_f2:.4f}")
    print(f"  Recall (Sensitivity): {oof_recall:.4f}")
    print(f"  Precision           : {oof_prec:.4f}")
    _print_separator()

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(oof_true, oof_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n  OOF CONFUSION MATRIX")
    print("  " + "─" * 40)
    print("                  Predicted NOT Churn   Predicted CHURN")
    print(f"  Actual NOT Churn       {tn:>8,}            {fp:>8,}")
    print(f"  Actual CHURN           {fn:>8,}            {tp:>8,}")
    print("  " + "─" * 40)
    print(f"  Churners caught  : {tp}/{tp+fn}  ({tp/(tp+fn)*100:.1f}%)")
    print(f"  False alarms (FP): {fp:,}")
    print(f"  Missed churners  : {fn}")
    print()

    # ── Classification Report ─────────────────────────────────────────────────
    print("  CLASSIFICATION REPORT  (threshold = {:.4f})".format(best_threshold))
    print("  " + "─" * 40)
    print(classification_report(
        oof_true, oof_pred,
        target_names=["Non-Churn (0)", "Churn (1)"],
        zero_division=0,
    ))

    # ── F2 Threshold Sweep (top 10 rows near optimal) ────────────────────────
    best_idx    = sweep_df["f2"].idxmax()
    context_idx = list(range(max(0, best_idx - 4), min(len(sweep_df), best_idx + 6)))
    print("  F2-SWEEP SNAPSHOT (rows closest to optimal threshold)")
    print("  " + "─" * 40)
    print(sweep_df.iloc[context_idx].to_string(index=False))
    print()

    # ── Feature Importance Plot ───────────────────────────────────────────────
    fi_df = pd.DataFrame({
        "feature"   : feature_names,
        "importance": fold_importances,
    }).sort_values("importance", ascending=False).head(15)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Telecom Churn XGBoost — Diagnostic Dashboard", fontsize=14, fontweight="bold", y=0.98)

    # Left panel: Top-15 feature importance
    ax_fi = axes[0]
    colors = sns.color_palette("RdYlGn_r", n_colors=15)
    bars = ax_fi.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color=colors, edgecolor="white", linewidth=0.6)
    ax_fi.set_title("Top 15 Feature Importances (Mean across 5 Folds)", fontsize=11, pad=10)
    ax_fi.set_xlabel("Mean XGBoost Feature Importance", fontsize=10)
    ax_fi.tick_params(axis="y", labelsize=8)
    ax_fi.tick_params(axis="x", labelsize=9)
    for bar in bars:
        w = bar.get_width()
        ax_fi.text(w + 0.0005, bar.get_y() + bar.get_height() / 2,
                   f"{w:.4f}", va="center", ha="left", fontsize=7)
    ax_fi.spines[["top", "right"]].set_visible(False)
    ax_fi.set_xlim(right=fi_df["importance"].max() * 1.20)

    # Right panel: F2 & Recall vs Threshold curve
    ax_th = axes[1]
    ax_th.plot(sweep_df["threshold"], sweep_df["f2"],
               label="F2 Score", color="#E34234", linewidth=2.5)
    ax_th.plot(sweep_df["threshold"], sweep_df["recall"],
               label="Recall", color="#2E86AB", linewidth=2.0, linestyle="--")
    ax_th.plot(sweep_df["threshold"], sweep_df["precision"],
               label="Precision", color="#57A773", linewidth=2.0, linestyle=":")
    ax_th.axvline(best_threshold, color="black", linestyle="-.", linewidth=1.5,
                  label=f"Optimal threshold = {best_threshold:.4f}")
    ax_th.set_title("F2 / Recall / Precision vs. Classification Threshold", fontsize=11, pad=10)
    ax_th.set_xlabel("Classification Threshold", fontsize=10)
    ax_th.set_ylabel("Score", fontsize=10)
    ax_th.legend(fontsize=9)
    ax_th.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_th.spines[["top", "right"]].set_visible(False)
    ax_th.set_ylim(0, 1.05)
    ax_th.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "churn_model_dashboard.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Dashboard saved → %s", plot_path)

    # ── Save sweep CSV ────────────────────────────────────────────────────────
    sweep_path = os.path.join(output_dir, "threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    log.info("Threshold sweep saved → %s", sweep_path)

    # ── Save feature importance CSV ───────────────────────────────────────────
    fi_full = pd.DataFrame({
        "feature"   : feature_names,
        "importance": fold_importances,
    }).sort_values("importance", ascending=False)
    fi_path = os.path.join(output_dir, "feature_importances.csv")
    fi_full.to_csv(fi_path, index=False)
    log.info("Feature importances saved → %s", fi_path)


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _print_separator("═", 72)
    print("   CHURN PREDICTION PIPELINE  —  Behavioral XGBoost Model")
    _print_separator("═", 72)

    # ── Step 1: Load raw data ─────────────────────────────────────────────────
    raw_df = load_raw_data()

    # Ensure target column present
    target = CONFIG["TARGET"]
    if target not in raw_df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {list(raw_df.columns[:10])} …")

    raw_df[target] = pd.to_numeric(raw_df[target], errors="coerce").fillna(0).astype(int)
    churn_rate = raw_df[target].mean()
    log.info("Target column '%s' | churn rate = %.4f%%", target, churn_rate * 100)

    # ── Step 2: Feature engineering ──────────────────────────────────────────
    feature_df = engineer_features(raw_df)

    # ── Step 3: Preprocessing ────────────────────────────────────────────────
    X, y, feature_names, scaler = preprocess(feature_df)

    # ── Step 4: Stratified cross-validation ──────────────────────────────────
    oof_prob, oof_true, fold_importances = run_stratified_cv(X, y, feature_names)

    # ── Step 5: Threshold optimisation ───────────────────────────────────────
    best_threshold, sweep_df = optimise_threshold(
        oof_true, oof_prob,
        th_min=CONFIG["THRESHOLD_MIN"],
        th_max=CONFIG["THRESHOLD_MAX"],
        steps =CONFIG["THRESHOLD_STEPS"],
    )

    # ── Step 6: Full evaluation report ───────────────────────────────────────
    report_metrics(
        oof_true, oof_prob, best_threshold, sweep_df,
        feature_names, fold_importances,
        output_dir=CONFIG["OUTPUT_DIR"],
    )

    _print_separator()
    print("  Pipeline complete. Outputs written to:", CONFIG["OUTPUT_DIR"])
    _print_separator()


if __name__ == "__main__":
    main()
