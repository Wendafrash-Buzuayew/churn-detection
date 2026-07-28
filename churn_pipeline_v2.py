"""
churn_pipeline_v2.py  ·  v3  (Cost-Sensitive + Rank-Blend Edition)
====================================================================
3-model stacked ensemble for telecom churn under extreme class imbalance
(~0.5% base churn rate, ~3.8M daily subscribers).

v3 changes — addressing four root-cause problems:

  PROBLEM 1 — SYNTHETIC FP FLOODING FROM SMOTE
    Root cause: Borderline-SMOTE on sparse zero-inflated telecom metrics
    (e.g. DATA_MB_W10=0 for 85% of subscribers) creates non-physical synthetic
    records that push the minority manifold into the majority region. The model
    learns these phantoms and floods false positives at inference time.
    Fix: SMOTE entirely removed. Class imbalance handled via cost-sensitive
    learning: `scale_pos_weight = num_negatives / num_positives` in XGBoost;
    `class_weight="balanced"` in RF/ET (unchanged — already correct).

  PROBLEM 2 — PROBABILITY CALIBRATION SKEW IN META-LEARNER
    Root cause: Logistic Regression stacking on raw model probabilities fails
    under extreme imbalance. The meta-learner collapses to a near-zero
    probability regime (all outputs < 0.02), making the threshold sweep
    operate on a degenerate scale.
    Fix: Percentile Rank Blending via scipy.stats.rankdata. Each model's OOF
    predictions are converted to normalized ranks [0, 1] before averaging.
    This makes the ensemble output scale-invariant and calibration-agnostic —
    the relative ordering is preserved regardless of probability distortion.

  PROBLEM 3 — THRESHOLD SWEEP MISSING THE PRECISION MANDATE
    Root cause: The original precision floor of 0.05 is ineffective at 0.5%
    base rate (0.05 ≈ 9× lift is still a flood at 3.8M scale). The fallback
    path (unconstrained F2-max) gravitated toward the lowest threshold.
    Fix: `optimize_threshold_precision_first()` sweeps 500 steps from 0.01
    to 0.99 and returns the highest F2 *strictly constrained* by precision
    >= min_precision (default 0.15). If no threshold clears the floor, the
    highest-achievable-precision threshold is returned with a loud warning.

  PROBLEM 4 — 4-WEEK AVERAGE MASKING W13 COLLAPSE (FN ROOT CAUSE)
    Root cause: Features like DATA_MB_RECENT_4W average W10–W13 together,
    keeping a healthy-looking score even when W13 has already terminated.
    Fix: Explicit W13 ratio features: W13 / mean(W10..W12) for Data MB,
    Voice Min, SMS Count, Bundle Count. Also added
    `consecutive_zero_usage_weeks` per service — the maximum run of
    consecutive zero weeks in the 4W window.

Run:
    INPUT_CSV=Feb_Train.csv python churn_pipeline_v2.py
    RUN_MODE=ORACLE python churn_pipeline_v2.py
    MODE=SCORE RUN_MODE=CSV INPUT_CSV=<file> python churn_pipeline_v2.py

Output (./churn_v2_outputs/):
    oof_threshold_sweep.csv        — threshold sweep table (OOF)
    test_threshold_sweep.csv       — threshold sweep table (TEST)
    feature_importance.csv         — per-model MI + importance ranking
    churn_model_artifacts.joblib   — serialised models for batch scoring
    churn_v2_dashboard.png         — 5-panel diagnostic dashboard
    production_churn_scores.csv    — (SCORE mode) full scored population
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import os, sys, warnings, logging, time
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.stats import rankdata         # rank-based blending (v3)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, fbeta_score,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    _XGB_OK = True
except ImportError:
    _XGB_OK = False
    log.error("XGBoost missing — install with: pip install xgboost")

try:
    import joblib
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False
    log.warning("joblib missing — model serialisation disabled")


# ── 1. Configuration ──────────────────────────────────────────────────────────

CHURN_HORIZON: int = int(os.getenv("CHURN_HORIZON", "30"))
_TARGET_COL:   str = f"LABEL_CHURN_{CHURN_HORIZON}D"

CFG = {
    # ── Data source ───────────────────────────────────────────────────────────
    "RUN_MODE"        : os.getenv("RUN_MODE", "CSV").upper(),
    "INPUT_CSV"       : os.getenv("INPUT_CSV", "Feb_Train.csv"),
    # ── Oracle connection (all values overridable via env vars) ───────────────
    "ORA_HOST"        : os.getenv("ORA_HOST",    "mdc1-charli-scan.safaricomet.net"),
    "ORA_PORT"        : int(os.getenv("ORA_PORT", "1521")),
    "ORA_SERVICE"     : os.getenv("ORA_SERVICE",  "DMCVLIVE.safaricomet.net"),
    "ORA_USER"        : os.getenv("ORA_USER",     "CVM_DM_PROD"),
    "ORA_PASSWORD"    : os.getenv("ORA_PASSWORD", ""),
    "ORA_TABLE"       : os.getenv("ORA_TABLE",
                        "CVM_DM_PROD.CHURN_POC_JAN15_FULL_FEATURES_V2"),
    "ORA_FETCH_CHUNK" : int(os.getenv("ORA_FETCH_CHUNK", "50000")),
    "ORA_SAMPLE_PCT"  : float(os.getenv("ORA_SAMPLE_PCT", "100")),
    # ── Prediction horizon ────────────────────────────────────────────────────
    "CHURN_HORIZON"   : CHURN_HORIZON,
    "TARGET"          : _TARGET_COL,
    "TARGET_FALLBACK" : "LABEL_CHURN_90D",
    # ── General ───────────────────────────────────────────────────────────────
    "DATASET_TYPE_COL": "DATASET_TYPE",
    "OUTPUT_DIR"      : "./churn_v2_outputs",
    "RANDOM_STATE"    : 42,
    "N_FOLDS"         : 5,
    # ── Feature selection ─────────────────────────────────────────────────────
    "TOP_K_FEATURES"  : 70,
    "WINSOR_P_LOW"    : 0.01,
    "WINSOR_P_HIGH"   : 0.99,
    # ── Rank-blend weights (XGB, RF, ET) ─────────────────────────────────────
    # Weights are updated automatically after OOF evaluation (proportional to
    # each model's OOF ROC-AUC). Set equal weights here as the initialisation.
    "BLEND_WEIGHTS"   : [1/3, 1/3, 1/3],
    # ── Threshold sweep (v3: precision-first) ─────────────────────────────────
    "TH_MIN"          : 0.01,
    "TH_MAX"          : 0.99,
    "TH_STEPS"        : 500,           # finer than v2 (was 400)
    "PRECISION_FLOOR" : float(os.getenv("PRECISION_FLOOR", "0.15")),
    # ── Alert-budget operating points ─────────────────────────────────────────
    "ALERT_BUDGET_PCTS": [0.0025, 0.005, 0.01, 0.02, 0.05],
    # ── Production scoring ────────────────────────────────────────────────────
    "SCORE_CHUNK_SIZE" : int(os.getenv("SCORE_CHUNK_SIZE", "200000")),
    "MODEL_DIR"        : os.getenv("MODEL_DIR", "./churn_v2_outputs/model_artifacts"),
    "SCORE_DTYPE"      : np.float32,
    # ── XGBoost (cost-sensitive, no SMOTE) ───────────────────────────────────
    # scale_pos_weight is set DYNAMICALLY per fold from class counts.
    # It is NOT fixed here — see build_xgb() and run_stacked_cv().
    "XGB": {
        "n_estimators"    : 800,
        "max_depth"       : 3,
        "learning_rate"   : 0.02,
        "subsample"       : 0.75,
        "colsample_bytree": 0.70,
        "min_child_weight": 5,
        "reg_lambda"      : 15.0,
        "reg_alpha"       : 2.0,
        "tree_method"     : "hist",
        "objective"       : "binary:logistic",
        "eval_metric"     : "aucpr",
        "n_jobs"          : -1,
    },
    # ── Random Forest (class_weight="balanced" — unchanged from v2) ───────────
    "RF": {
        "n_estimators"   : 500,
        "max_depth"      : 6,
        "min_samples_leaf": 5,
        "max_features"   : "sqrt",
        "n_jobs"         : -1,
    },
    # ── Extra Trees (class_weight="balanced") ─────────────────────────────────
    "ET": {
        "n_estimators"   : 500,
        "max_depth"      : 6,
        "min_samples_leaf": 5,
        "max_features"   : "sqrt",
        "n_jobs"         : -1,
    },
}

os.makedirs(CFG["OUTPUT_DIR"], exist_ok=True)
os.makedirs(CFG["MODEL_DIR"],  exist_ok=True)

# ── Columns excluded from modelling ──────────────────────────────────────────
_ID_COLS = {
    "MSISDN", "MSISDN_9", "MSISDN_251", "SNAPSHOT_DATE",
    "AON",        # kept separately — added back as numeric feature
    "DATASET_TYPE",
    "LABEL_CHURN_30D",
    "LABEL_CHURN_90D",
}

# Highly correlated duplicates (r > 0.98 on training data)
_DROP_REDUNDANT = {
    "TOTAL_VOICE_MIN_W10","TOTAL_VOICE_MIN_W11","TOTAL_VOICE_MIN_W12","TOTAL_VOICE_MIN_W13",
    "TOTAL_VOICE_MIN_RECENT_4W",
    "TOTAL_VOICE_MIN_TREND_SLOPE_13W","TOTAL_VOICE_MIN_VOLATILITY_13W",
    "TOTAL_VOICE_MIN_CV_13W","TOTAL_VOICE_MIN_ZERO_WEEKS_13W",
    "TOTAL_VOICE_MIN_ZERO_WEEKS_RECENT_4W",
    "DATA_MB_ZERO_WEEKS_RECENT_4W",
    "TOTAL_SMS_COUNT_ZERO_WEEKS_RECENT_4W",
    "BUNDLE_CNT_ZERO_WEEKS_RECENT_4W",
}

# Keywords that indicate extreme-valued columns needing winsorisation
_WINSOR_COLS_KEYWORDS = ["_MB", "_MIN", "_CNT", "_REV", "AON", "_DAYS", "SLOPE",
                          "VOLATIL", "_PEAK", "_DROP"]

# Force these engineered features past the MI cutoff (multi-column interactions)
FORCE_INCLUDE_FEATURES = [
    "ALL_SVC_ZERO_W13_FLAG",
    "ENGAGEMENT_IDX",
    "SHORT_TENURE_FLAG",
    "MULTI_SVC_ZERO_W13",
    # v3 W13 ratio features (always include — they encode the FN root cause)
    "W13_RATIO_DATA_MB",
    "W13_RATIO_VOICE_MIN",
    "W13_RATIO_SMS_COUNT",
    "W13_RATIO_BUNDLE_CNT",
    "CONSEC_ZERO_WEEKS_DATA",
    "CONSEC_ZERO_WEEKS_VOICE",
    "CONSEC_ZERO_WEEKS_SMS",
    "CONSEC_ZERO_WEEKS_BUNDLE",
    "MAX_CONSEC_ZERO_ANY_SVC",
]


# ── 2. Feature Engineering ────────────────────────────────────────────────────

def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    """Safely fetch a column; return `default` constant series if absent."""
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def _w13_ratio(df: pd.DataFrame, prefix: str) -> pd.Series:
    """
    Compute W13 / mean(W10, W11, W12) for a given service prefix.

    This ratio is the primary FN fix: it is exactly 1.0 if the subscriber's
    W13 activity equals their prior 3-week average, → 0 if W13 has collapsed,
    and > 1 if W13 has accelerated. Unlike the 4-week aggregate features,
    this is NOT masked by earlier high-activity weeks.

    Safe division: where mean(W10, W11, W12) == 0, the ratio is set to 0.0
    (subscriber was already inactive — a ratio of 0 correctly signals no
    activity, consistent with the churner profile).
    """
    w10 = pd.to_numeric(_col(df, f"{prefix}_W10"), errors="coerce").fillna(0.0)
    w11 = pd.to_numeric(_col(df, f"{prefix}_W11"), errors="coerce").fillna(0.0)
    w12 = pd.to_numeric(_col(df, f"{prefix}_W12"), errors="coerce").fillna(0.0)
    w13 = pd.to_numeric(_col(df, f"{prefix}_W13"), errors="coerce").fillna(0.0)
    baseline = (w10 + w11 + w12) / 3.0
    ratio = np.where(baseline > 0.0, w13 / baseline, 0.0)
    return pd.Series(ratio, index=df.index, dtype=np.float32)


def _consecutive_zero_weeks(df: pd.DataFrame, prefix: str) -> pd.Series:
    """
    Count the maximum run of consecutive zero-usage weeks in [W10, W11, W12, W13].

    Returns an integer series 0–4:
      0 = active every week
      4 = no activity in any of the last 4 weeks (dead SIM)

    This feature is strictly more informative than the simple 4-week aggregate:
    a subscriber with pattern [100, 0, 0, 0] has 3 consecutive zeros and is
    likely churning, whereas [0, 100, 0, 100] has max 1 consecutive zero and
    is just intermittent.
    """
    weeks = []
    for w in ["W10", "W11", "W12", "W13"]:
        col_name = f"{prefix}_{w}"
        v = pd.to_numeric(_col(df, col_name), errors="coerce").fillna(0.0)
        weeks.append((v <= 0).astype(int))

    # Stack into (n, 4) matrix of binary zero-indicators
    mat = np.column_stack([s.values for s in weeks])  # shape (n, 4)

    # Compute maximum consecutive run of 1s (zeros) per row efficiently
    def _max_run(row):
        """Maximum run-length of 1s in a binary array."""
        max_r = cur_r = 0
        for v in row:
            cur_r = cur_r + 1 if v else 0
            max_r = max(max_r, cur_r)
        return max_r

    runs = np.apply_along_axis(_max_run, 1, mat)
    return pd.Series(runs, index=df.index, dtype=np.int8)


def engineer_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction / composite features.

    v3 additions (W13 ratio features + consecutive zero weeks):
    ────────────────────────────────────────────────────────────
    W13_RATIO_DATA_MB      : DATA_MB_W13 / mean(DATA_MB_W10..W12)
    W13_RATIO_VOICE_MIN    : OG_VOICE_MIN_W13 / mean(OG_VOICE_MIN_W10..W12)
    W13_RATIO_SMS_COUNT    : OG_SMS_COUNT_W13 / mean(OG_SMS_COUNT_W10..W12)
    W13_RATIO_BUNDLE_CNT   : BUNDLE_CNT_W13 / mean(BUNDLE_CNT_W10..W12)
      → Ratio < 0.15 means W13 collapsed to <15% of prior baseline.
        The 4-week aggregate hides this; the ratio surfaces it directly.
        These features directly address the FN root cause identified in
        SHAP analysis (DATA_REVENUE_RECENT_4W as a safety driver).

    CONSEC_ZERO_WEEKS_DATA   : max consecutive zero-data weeks [W10..W13]
    CONSEC_ZERO_WEEKS_VOICE  : max consecutive zero-voice weeks
    CONSEC_ZERO_WEEKS_SMS    : max consecutive zero-SMS weeks
    CONSEC_ZERO_WEEKS_BUNDLE : max consecutive zero-bundle weeks
    MAX_CONSEC_ZERO_ANY_SVC  : max of all four consecutive-zero counts

    v2 features (retained unchanged):
    ───────────────────────────────────
    MULTI_SVC_ZERO_W13, ALL_SVC_ZERO_W13_FLAG, TOTAL_ZERO_WEEKS_ALL,
    REV_PER_ACTIVE_WEEK, SIMULTANEOUS_LONG_DROP, MAX_CONSEC_ZERO_ANY,
    ENGAGEMENT_IDX, AON_LOG, SHORT_TENURE_FLAG, AON_BUCKET,
    BUNDLE_REV_COLLAPSE
    """
    # ── v3 NEW: W13 acceleration-collapse ratio features ─────────────────────
    #
    # WHY RATIOS INSTEAD OF RAW W13:
    #   Raw W13 values are scale-dependent — a W13 of 10 MB could be a massive
    #   drop for a heavy user (baseline 500 MB) or normal for a light user.
    #   The ratio normalises by the subscriber's own prior baseline, making
    #   the collapse signal subscriber-agnostic and scale-invariant.
    #   A ratio close to 0 always means "this subscriber's last week was
    #   essentially dead relative to their recent habit", regardless of
    #   whether they use 1 MB or 1 GB per week.
    df["W13_RATIO_DATA_MB"]    = _w13_ratio(df, "DATA_MB")
    df["W13_RATIO_VOICE_MIN"]  = _w13_ratio(df, "OG_VOICE_MIN")
    df["W13_RATIO_SMS_COUNT"]  = _w13_ratio(df, "OG_SMS_COUNT")
    df["W13_RATIO_BUNDLE_CNT"] = _w13_ratio(df, "BUNDLE_CNT")

    # ── v3 NEW: Consecutive zero-usage weeks per service ─────────────────────
    #
    # WHY CONSECUTIVE ZEROS:
    #   A subscriber with [100, 0, 0, 0] is very different from [0, 100, 0, 100].
    #   The first has 3 consecutive terminal zeros — strong churn signal.
    #   The second is just intermittent — not a churner.
    #   Counting consecutive zeros separates these patterns; simple sum-of-zeros
    #   would score them equally (both have 2 zero weeks).
    df["CONSEC_ZERO_WEEKS_DATA"]   = _consecutive_zero_weeks(df, "DATA_MB")
    df["CONSEC_ZERO_WEEKS_VOICE"]  = _consecutive_zero_weeks(df, "OG_VOICE_MIN")
    df["CONSEC_ZERO_WEEKS_SMS"]    = _consecutive_zero_weeks(df, "OG_SMS_COUNT")
    df["CONSEC_ZERO_WEEKS_BUNDLE"] = _consecutive_zero_weeks(df, "BUNDLE_CNT")
    # Composite: worst-case across all services (any service hitting 4 = dead SIM)
    df["MAX_CONSEC_ZERO_ANY_SVC"] = df[[
        "CONSEC_ZERO_WEEKS_DATA", "CONSEC_ZERO_WEEKS_VOICE",
        "CONSEC_ZERO_WEEKS_SMS",  "CONSEC_ZERO_WEEKS_BUNDLE",
    ]].max(axis=1)

    # ── v2 features (unchanged) ───────────────────────────────────────────────

    # Multi-service simultaneous dead-week signal
    z_data   = (_col(df, "DATA_MB_W13")      <= 0).astype(int)
    z_voice  = (_col(df, "OG_VOICE_MIN_W13") <= 0).astype(int)
    z_sms    = (_col(df, "OG_SMS_COUNT_W13") <= 0).astype(int)
    z_bundle = (_col(df, "BUNDLE_CNT_W13")   <= 0).astype(int)
    multi_zero = z_data + z_voice + z_sms + z_bundle
    df["MULTI_SVC_ZERO_W13"]    = multi_zero
    df["ALL_SVC_ZERO_W13_FLAG"] = (multi_zero >= 4).astype(int)

    # Breadth of inactivity across the full 13-week window
    zw_cols = [c for c in df.columns if c.endswith("ZERO_WEEKS_13W")]
    df["TOTAL_ZERO_WEEKS_ALL"] = df[zw_cols].sum(axis=1) if zw_cols else 0.0

    # Revenue efficiency: revenue per active week
    df["REV_PER_ACTIVE_WEEK"] = _col(df, "TOTAL_REVENUE_RECENT_4W") / (
        _col(df, "DATA_ACTIVE_WEEKS_RECENT_4W") + 1.0
    )

    # How many services show a positive long-run drop simultaneously
    long_drop_cols = [c for c in df.columns if c.endswith("LONG_DROP_PCT")]
    if long_drop_cols:
        ld = df[long_drop_cols].clip(-1000, 1000)
        df["SIMULTANEOUS_LONG_DROP"] = (ld > 0).sum(axis=1)
    else:
        df["SIMULTANEOUS_LONG_DROP"] = 0.0

    # Deepest single-service consecutive zero streak (v2 version — max 13W)
    consec_cols_13w = [c for c in df.columns if c.endswith("CONSECUTIVE_ZERO_WEEKS_RECENT")]
    df["MAX_CONSEC_ZERO_ANY"] = df[consec_cols_13w].max(axis=1) if consec_cols_13w else 0.0

    # Composite engagement index
    active_sum = (
        _col(df, "DATA_ACTIVE_WEEKS_RECENT_4W") +
        _col(df, "TOTAL_VOICE_ACTIVE_WEEKS_RECENT_4W") +
        _col(df, "TOTAL_SMS_ACTIVE_WEEKS_RECENT_4W") +
        _col(df, "BUNDLE_ACTIVE_WEEKS_RECENT_4W")
    )
    df["ENGAGEMENT_IDX"] = active_sum * _col(df, "SERVICE_DIVERSITY_RECENT_4W") / 64.0

    # Tenure signals
    aon = _col(df, "AON")
    df["AON_LOG"]           = np.log1p(aon.clip(lower=0))
    df["SHORT_TENURE_FLAG"] = (aon < 180).astype(int)
    df["AON_BUCKET"] = pd.cut(
        aon, bins=[-1, 90, 180, 365, 730, 1e9], labels=[4, 3, 2, 1, 0]
    ).astype(float)

    # Bundle abandonment × revenue collapse
    df["BUNDLE_REV_COLLAPSE"] = (
        _col(df, "BUNDLE_CNT_PEAK_TO_RECENT_DROP_PCT") *
        _col(df, "TOTAL_REVENUE_LONG_DROP_PCT").clip(-10, 10)
    )

    return df


# ── 3. Data loading ───────────────────────────────────────────────────────────

def _resolve_target(df: pd.DataFrame) -> str:
    """Resolve the churn label column name from CHURN_HORIZON setting."""
    primary  = CFG["TARGET"]
    fallback = CFG["TARGET_FALLBACK"]
    for col in [primary, fallback]:
        if col in df.columns:
            if col != primary:
                log.warning("'%s' not found — using '%s'", primary, col)
            return col
    hits = [c for c in df.columns if c.startswith("LABEL_CHURN_")]
    if hits:
        log.warning("Using first available churn label: '%s'", hits[0])
        return hits[0]
    raise ValueError(
        f"No churn label column found. Expected '{primary}' or '{fallback}'."
    )


def _fetch_oracle() -> pd.DataFrame:
    """Stream from Oracle CVM_DM_PROD in memory-safe chunks."""
    try:
        import oracledb
    except ImportError:
        raise RuntimeError("oracledb not installed. Run: pip install oracledb")

    conn = oracledb.connect(
        user=CFG["ORA_USER"], password=CFG["ORA_PASSWORD"],
        dsn=f"{CFG['ORA_HOST']}:{CFG['ORA_PORT']}/{CFG['ORA_SERVICE']}",
    )
    sample_clause = (
        f"SAMPLE({CFG['ORA_SAMPLE_PCT']})" if CFG["ORA_SAMPLE_PCT"] < 100 else ""
    )
    sql = f"SELECT * FROM {CFG['ORA_TABLE']} {sample_clause}"
    log.info("Fetching from Oracle: %s", sql[:120])

    chunks = []
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0].upper() for d in cur.description]
        while True:
            rows = cur.fetchmany(CFG["ORA_FETCH_CHUNK"])
            if not rows:
                break
            chunks.append(pd.DataFrame(rows, columns=cols))
            log.info("  … fetched %d rows", sum(len(c) for c in chunks))

    conn.close()
    df = pd.concat(chunks, ignore_index=True)
    log.info("Oracle load complete: %d rows × %d cols", *df.shape)
    return df


def load_and_clean() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Dual-mode data loader: Oracle DB or CSV flat file.

    Returns (train_df, test_df, feature_cols).
    DATASET_TYPE column (values: TRAIN / TEST / OOT) drives the split;
    if absent, a stratified 70/30 random split is used.
    """
    mode = CFG["RUN_MODE"]
    if mode == "ORACLE":
        log.info("Data source: ORACLE  (horizon: %dD)", CFG["CHURN_HORIZON"])
        df = _fetch_oracle()
    else:
        log.info("Data source: CSV  (horizon: %dD)", CFG["CHURN_HORIZON"])
        path = CFG["INPUT_CSV"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"CSV not found: {path}\n"
                f"Set INPUT_CSV=/path/to/file.csv  or  RUN_MODE=ORACLE"
            )
        df = pd.read_csv(path)
        df.columns = [c.upper().strip() for c in df.columns]
        log.info("CSV loaded: %d rows × %d cols", *df.shape)

    # Resolve and pin target
    target      = _resolve_target(df)
    CFG["TARGET"] = target
    log.info("Target column: %s  (CHURN_HORIZON=%dD)", target, CFG["CHURN_HORIZON"])
    df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)

    # AON as numeric feature
    if "AON" in df.columns:
        df["AON"] = pd.to_numeric(df["AON"], errors="coerce").fillna(0)

    # Engineer features (v3: adds W13 ratios + consecutive-zero weeks)
    n_cols_before = df.shape[1]
    df = engineer_extra_features(df)
    log.info("Engineered %d new features (v3 includes W13 ratios + consec-zero)",
             df.shape[1] - n_cols_before)

    # Feature column list
    feat_cols = [
        c for c in df.columns
        if c not in _ID_COLS and c not in _DROP_REDUNDANT and c != target
    ]
    if "AON" in df.columns:
        feat_cols = ["AON"] + [c for c in feat_cols if c != "AON"]

    # Numeric cast + infinity removal
    for col in feat_cols:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
              .replace([np.inf, -np.inf], np.nan)
              .fillna(0.0)
        )

    # Winsorise extreme-valued columns
    for col in feat_cols:
        if any(kw in col for kw in _WINSOR_COLS_KEYWORDS):
            lo = df[col].quantile(CFG["WINSOR_P_LOW"])
            hi = df[col].quantile(CFG["WINSOR_P_HIGH"])
            df[col] = df[col].clip(lo, hi)

    # TRAIN / TEST split logic
    ds_col = CFG["DATASET_TYPE_COL"]
    
    # Check if column exists and contains actual TRAIN rows
    if ds_col in df.columns and (df[ds_col].astype(str).str.upper() == "TRAIN").any():
        train_df = df[df[ds_col].astype(str).str.upper() == "TRAIN"].reset_index(drop=True)
        test_df  = df[df[ds_col].astype(str).str.upper() == "TEST"].reset_index(drop=True)
        
        # If TEST is empty, pull TEST from OOT if present, or split TRAIN
        if len(test_df) == 0:
            log.warning("No explicit TEST rows found. Checking for OOT rows as test set...")
            test_df = df[df[ds_col].astype(str).str.upper() == "OOT"].reset_index(drop=True)
            
        log.info("Partition via %s: TRAIN=%d rows, TEST=%d rows",
                 ds_col, len(train_df), len(test_df))
    else:
        # Triggers when column is missing OR when all values are 'OOT' / non-TRAIN
        if ds_col in df.columns:
            unique_vals = df[ds_col].astype(str).str.upper().unique().tolist()
            log.warning("'%s' column found with values %s but NO 'TRAIN' rows detected. "
                        "Falling back to 70/30 stratified split on available data.", ds_col, unique_vals)
        else:
            log.warning("No '%s' column found — auto-splitting 70/30 stratified.", ds_col)

        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=0.30, stratify=df[target],
            random_state=CFG["RANDOM_STATE"]
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    if len(train_df) == 0:
        raise RuntimeError(
            f"TRAIN set is empty. Input dataset shape is {df.shape}. "
            "Please verify dataset labels or input file."
        )

    return train_df, test_df, feat_cols


# ── 4. Feature selection ──────────────────────────────────────────────────────

def select_features(
    train_df  : pd.DataFrame,
    feat_cols : List[str],
    top_k     : int,
) -> Tuple[List[str], np.ndarray]:
    """
    Mutual Information feature selection on training data.
    FORCE_INCLUDE_FEATURES are kept regardless of MI rank (they encode
    multi-column interactions that MI under-represents individually).
    """
    X_tr = train_df[feat_cols].values.astype(float)
    y_tr = train_df[CFG["TARGET"]].values

    scaler = RobustScaler()
    X_s    = scaler.fit_transform(X_tr)

    log.info("Computing Mutual Information on %d features × %d rows …",
             len(feat_cols), len(y_tr))
    mi        = mutual_info_classif(X_s, y_tr, random_state=CFG["RANDOM_STATE"],
                                    n_neighbors=3)
    mi_series = pd.Series(mi, index=feat_cols).sort_values(ascending=False)
    selected  = mi_series.head(top_k).index.tolist()

    forced = [f for f in FORCE_INCLUDE_FEATURES
              if f in feat_cols and f not in selected]
    if forced:
        log.info("Force-including %d v3 features past MI cutoff: %s", len(forced), forced)
        selected = selected + forced

    log.info("Selected %d features (top-%d MI + %d forced)",
             len(selected), top_k, len(forced))
    return selected, mi


# ── 5. Model builders ─────────────────────────────────────────────────────────

def build_xgb(scale_pos_weight: float) -> "XGBClassifier":
    """
    XGBoost with dynamically computed scale_pos_weight.

    v3 change: SMOTE is completely removed. Class imbalance is handled here
    via scale_pos_weight = neg_count / pos_count (computed per fold or on
    the full training set). This is XGBoost's native cost-sensitive mechanism:
    the gradient update for each positive sample is multiplied by this weight,
    giving the minority class proportionally more influence on the loss.

    WHY scale_pos_weight BEATS SMOTE HERE:
      • SMOTE creates synthetic interpolations in feature space. For sparse
        zero-inflated telecom data (most features are exactly 0 for 85%+ of
        subscribers), interpolation between two zero vectors produces another
        zero vector — completely non-informative. Interpolation between a
        zero vector and a non-zero vector produces an intermediate that does
        not correspond to any real subscriber behaviour.
      • scale_pos_weight operates on the actual minority samples without
        modifying the data distribution. It's mathematically equivalent to
        repeating each minority sample scale_pos_weight times in the loss
        function, but without the memory or manifold distortion of SMOTE.
    """
    params = {
        **CFG["XGB"],
        "scale_pos_weight": scale_pos_weight,
        "random_state"    : CFG["RANDOM_STATE"],
    }
    return XGBClassifier(**params)


def build_rf() -> RandomForestClassifier:
    """
    Random Forest with class_weight="balanced".
    balanced = n_samples / (n_classes × np.bincount(y)) per sample.
    This is the RF-native equivalent of scale_pos_weight for XGBoost.
    """
    return RandomForestClassifier(
        **CFG["RF"],
        class_weight="balanced",
        random_state=CFG["RANDOM_STATE"],
    )


def build_et() -> ExtraTreesClassifier:
    """Extra Trees with class_weight="balanced"."""
    return ExtraTreesClassifier(
        **CFG["ET"],
        class_weight="balanced",
        random_state=CFG["RANDOM_STATE"],
    )


# ── 6. Rank-Based Ensembling ──────────────────────────────────────────────────

def rank_blend(
    p_xgb        : np.ndarray,
    p_rf         : np.ndarray,
    p_et         : np.ndarray,
    weights      : Optional[List[float]] = None,
) -> np.ndarray:
    """
    Convert each model's probability predictions to normalised percentile
    ranks, then compute a weighted average of ranks.

    WHY RANK BLENDING INSTEAD OF LOGISTIC REGRESSION STACKING:

    Under extreme class imbalance (0.5% base rate), raw model probabilities
    suffer from calibration collapse: tree ensembles tend to produce outputs
    in a very narrow band (e.g. 0.001–0.02 for all subscribers), making the
    logistic regression meta-learner's weight estimation unstable. Small
    calibration differences between XGBoost and RF lead to large coefficient
    instabilities in the LR, effectively discarding one of the models.

    Rank-based blending sidesteps this entirely:
      1. Convert each model's output vector to fractional ranks in [0, 1].
         Rank i = (position of p_i when sorted) / n.
         This is scale-invariant and calibration-agnostic.
      2. Weighted-average the rank vectors.
         The weights can be uniform (1/3 each) or AUC-proportional (see
         run_stacked_cv for automatic AUC-based weight computation).

    The blended output is still in [0, 1] and preserves the relative ordering
    of subscribers (the most at-risk subscriber still gets rank ≈ 1.0).
    It is NOT a calibrated probability — it is a risk percentile index.
    The threshold optimiser treats it exactly the same way it treats a
    probability: it finds the cut-point that maximises F2 at precision ≥ 0.15.

    Parameters
    ──────────
    p_xgb, p_rf, p_et : 1-D arrays of model predictions (probabilities or scores).
    weights : list of three non-negative weights summing to > 0.
              If None, CFG["BLEND_WEIGHTS"] is used (default: [1/3, 1/3, 1/3]).

    Returns
    ───────
    blended : 1-D float32 array of normalised rank scores in [0, 1].
    """
    if weights is None:
        weights = CFG["BLEND_WEIGHTS"]

    n = len(p_xgb)
    if n == 0:
        return np.array([], dtype=np.float32)

    # scipy.stats.rankdata assigns average ranks to ties (method="average"),
    # which handles the common case of many subscribers with the same
    # score (e.g. 0.0 for the all-zero cluster). Dividing by n normalises
    # ranks to (0, 1].
    r_xgb = rankdata(p_xgb, method="average") / n
    r_rf  = rankdata(p_rf,  method="average") / n
    r_et  = rankdata(p_et,  method="average") / n

    w     = np.array(weights, dtype=np.float64)
    w     = w / w.sum()          # normalise so weights always sum to 1

    blended = w[0] * r_xgb + w[1] * r_rf + w[2] * r_et
    return blended.astype(np.float32)


# ── 7. 5-Fold Stacked CV (cost-sensitive, rank-blend) ─────────────────────────

def run_stacked_cv(
    train_df     : pd.DataFrame,
    selected_feat: List[str],
    all_feat_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray]:
    """
    5-Fold Stratified CV on the TRAIN partition.

    v3 changes vs v2:
      • No SMOTE: training folds are used as-is; XGBoost receives
        scale_pos_weight = neg_count / pos_count computed per fold.
      • No LR meta-learner: OOF predictions are rank-blended using
        AUC-proportional weights (higher AUC → higher blend weight).
      • Returns blend_weights so evaluate_on_test() and _score_batch()
        use the same weights as the OOF evaluation.

    Returns
    ───────
    oof_blend    : (n_train,) float32 — blended rank score per subscriber
    oof_true     : (n_train,) int     — ground truth labels
    fi_dict      : {'xgb': array, 'rf': array, 'et': array} feature importances
    blend_weights: (3,) float array   — AUC-proportional blend weights [XGB, RF, ET]
    """
    if not _XGB_OK:
        raise RuntimeError("XGBoost is required. pip install xgboost")

    X_all = train_df[selected_feat].values.astype(float)
    y_all = train_df[CFG["TARGET"]].values.astype(int)

    neg_total = int((y_all == 0).sum())
    pos_total = int((y_all == 1).sum())
    log.info("TRAIN  neg=%d  pos=%d  global_scale_pos_weight=%.1f",
             neg_total, pos_total, neg_total / max(pos_total, 1))

    skf = StratifiedKFold(
        n_splits=CFG["N_FOLDS"], shuffle=True, random_state=CFG["RANDOM_STATE"]
    )

    oof_xgb  = np.zeros(len(y_all), dtype=np.float64)
    oof_rf   = np.zeros(len(y_all), dtype=np.float64)
    oof_et   = np.zeros(len(y_all), dtype=np.float64)
    oof_true = np.zeros(len(y_all), dtype=int)
    fi_xgb   = np.zeros(len(selected_feat))
    fi_rf    = np.zeros(len(selected_feat))
    fi_et    = np.zeros(len(selected_feat))

    fold_auc_xgb = []
    fold_auc_rf  = []
    fold_auc_et  = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
        X_tr, X_val = X_all[tr_idx], X_all[val_idx]
        y_tr, y_val = y_all[tr_idx], y_all[val_idx]

        # ── Scale (fit on training fold only) ─────────────────────────────────
        scaler  = RobustScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # ── Compute fold-level scale_pos_weight ───────────────────────────────
        # Using the fold's actual class distribution (not the global ratio)
        # ensures each fold's XGBoost is calibrated to its local imbalance,
        # which can vary due to stratified sampling in small training sets.
        neg_fold = int((y_tr == 0).sum())
        pos_fold = int((y_tr == 1).sum())
        spw_fold = neg_fold / max(pos_fold, 1)

        log.info(
            "  Fold %d/%d │ train=%d (pos=%d, spw=%.1f) │ val=%d (pos=%d)",
            fold, CFG["N_FOLDS"],
            len(y_tr), pos_fold, spw_fold,
            len(y_val), int((y_val == 1).sum()),
        )

        # ── XGBoost (cost-sensitive, no SMOTE) ────────────────────────────────
        xgb_model = build_xgb(scale_pos_weight=spw_fold)
        xgb_model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
        p_xgb = xgb_model.predict_proba(X_val_s)[:, 1]

        # ── Random Forest (class_weight="balanced") ────────────────────────────
        rf_model = build_rf()
        rf_model.fit(X_tr_s, y_tr)
        p_rf = rf_model.predict_proba(X_val_s)[:, 1]

        # ── Extra Trees (class_weight="balanced") ──────────────────────────────
        et_model = build_et()
        et_model.fit(X_tr_s, y_tr)
        p_et = et_model.predict_proba(X_val_s)[:, 1]

        oof_xgb [val_idx] = p_xgb
        oof_rf  [val_idx] = p_rf
        oof_et  [val_idx] = p_et
        oof_true[val_idx] = y_val
        fi_xgb            += xgb_model.feature_importances_
        fi_rf             += rf_model.feature_importances_
        fi_et             += et_model.feature_importances_

        # Track per-fold AUC for blend weight computation
        try:
            auc_xgb_f = roc_auc_score(y_val, p_xgb)
            auc_rf_f  = roc_auc_score(y_val, p_rf)
            auc_et_f  = roc_auc_score(y_val, p_et)
            fold_auc_xgb.append(auc_xgb_f)
            fold_auc_rf.append(auc_rf_f)
            fold_auc_et.append(auc_et_f)
            log.info(
                "          Fold AUC → XGB=%.4f  RF=%.4f  ET=%.4f",
                auc_xgb_f, auc_rf_f, auc_et_f
            )
        except Exception:
            pass

    # ── AUC-proportional blend weights ────────────────────────────────────────
    # Weight each model by its average OOF AUC, normalised to sum to 1.
    # This automatically down-weights underperforming models without manual tuning.
    mean_auc_xgb = float(np.mean(fold_auc_xgb)) if fold_auc_xgb else 1/3
    mean_auc_rf  = float(np.mean(fold_auc_rf))  if fold_auc_rf  else 1/3
    mean_auc_et  = float(np.mean(fold_auc_et))  if fold_auc_et  else 1/3
    auc_sum      = mean_auc_xgb + mean_auc_rf + mean_auc_et
    blend_weights = np.array([
        mean_auc_xgb / auc_sum,
        mean_auc_rf  / auc_sum,
        mean_auc_et  / auc_sum,
    ], dtype=np.float64)
    CFG["BLEND_WEIGHTS"] = blend_weights.tolist()
    log.info(
        "AUC-proportional blend weights: XGB=%.3f  RF=%.3f  ET=%.3f",
        *blend_weights
    )

    # ── OOF rank blend ────────────────────────────────────────────────────────
    oof_blend = rank_blend(oof_xgb, oof_rf, oof_et, weights=blend_weights)

    log.info("OOF XGB ROC-AUC  = %.4f", roc_auc_score(oof_true, oof_xgb))
    log.info("OOF RF  ROC-AUC  = %.4f", roc_auc_score(oof_true, oof_rf))
    log.info("OOF ET  ROC-AUC  = %.4f", roc_auc_score(oof_true, oof_et))
    log.info("OOF Blend ROC-AUC = %.4f", roc_auc_score(oof_true, oof_blend))
    log.info("OOF Blend PR-AUC  = %.4f", average_precision_score(oof_true, oof_blend))

    fi_dict = {
        "xgb": fi_xgb / CFG["N_FOLDS"],
        "rf" : fi_rf  / CFG["N_FOLDS"],
        "et" : fi_et  / CFG["N_FOLDS"],
    }
    return oof_blend, oof_true, fi_dict, blend_weights


# ── 8. Precision-First Threshold Optimisation ─────────────────────────────────

def optimize_threshold_precision_first(
    y_true      : np.ndarray,
    y_probs     : np.ndarray,
    min_precision: float = 0.15,
    label       : str   = "OOF",
) -> Tuple[float, pd.DataFrame]:
    """
    Sweep thresholds from 0.01 to 0.99 (500 steps) and find the operating
    point with the highest F2 score *strictly constrained* by precision
    >= min_precision.

    DESIGN RATIONALE:
    ──────────────────
    At a 0.5% base rate and 3.8M subscribers, precision = 0.05 means
    95 false alarms per 5 true positives = 3.61M false alarms per day.
    That floods every downstream CRM system. Precision = 0.15 is the
    minimum level at which the alert list is economically actionable:
    15 true churners per 100 contacts is the lower bound for retention
    campaign ROI at typical telco treatment costs.

    F2 is used (not F1) because recall is more important than precision
    beyond the floor: missing a churner (FN) costs lifetime revenue,
    while an extra false alarm costs one treatment attempt. F2 weights
    recall twice as heavily as precision.

    FALLBACK BEHAVIOUR:
    ───────────────────
    If NO threshold achieves precision >= min_precision (possible in very
    small validation folds where the minority class is sparse), the function
    returns the threshold with the HIGHEST achievable precision and logs a
    prominent warning. It does NOT fall back to unconstrained F2-max, which
    at this base rate gravitates toward threshold → 0 and maximally floods
    false positives.

    Parameters
    ──────────
    y_true       : ground-truth binary labels.
    y_probs      : model scores or probabilities (rank-blend output or raw probs).
    min_precision: precision floor constraint. Default 0.15.
    label        : used in log messages for identification.

    Returns
    ───────
    best_th : float — the selected threshold.
    sweep   : pd.DataFrame — full sweep table (threshold, precision, recall,
              f1, f2, tp, fp, fn, tn, predicted_pos) for diagnostics.
    """
    thresholds = np.linspace(CFG["TH_MIN"], CFG["TH_MAX"], CFG["TH_STEPS"])
    rows = []

    for th in thresholds:
        pred  = (y_probs >= th).astype(int)
        cm    = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec  = precision_score(y_true, pred, zero_division=0)
        rec   = recall_score   (y_true, pred, zero_division=0)
        f1    = f1_score       (y_true, pred, zero_division=0)
        f2    = fbeta_score    (y_true, pred, beta=2, zero_division=0)
        rows.append(dict(
            threshold    = round(float(th), 4),
            precision    = round(prec,      4),
            recall       = round(rec,       4),
            f1           = round(f1,        4),
            f2           = round(f2,        4),
            tp           = int(tp),
            fp           = int(fp),
            fn           = int(fn),
            tn           = int(tn),
            predicted_pos= int(tp + fp),
        ))

    sweep   = pd.DataFrame(rows)
    floor   = min_precision
    guarded = sweep[sweep["precision"] >= floor]

    if guarded.empty:
        # No threshold clears the precision floor.
        # This typically happens when the model has poor discrimination on this
        # fold (e.g. very few positives in the validation set).
        best_achievable_prec = sweep["precision"].max()
        log.warning(
            "[%s] NO threshold reaches precision floor %.3f. "
            "Best achievable precision in this sweep = %.4f. "
            "Falling back to highest-precision threshold — NOT unconstrained "
            "F2-max (which would flood false alarms at 0.5%% base rate).",
            label, floor, best_achievable_prec,
        )
        guarded = sweep[sweep["precision"] == best_achievable_prec]

    # Primary selection: highest F2 within the precision-constrained set.
    # If multiple thresholds tie on F2 (e.g. all-zero F2 in a degenerate fold),
    # idxmax returns the first, which corresponds to the lowest threshold
    # (highest recall) within the constrained set — a reasonable fallback.
    best_row = guarded.loc[guarded["f2"].idxmax()]
    best_th  = float(best_row["threshold"])

    log.info(
        "[%s] Selected threshold=%.4f → F2=%.4f  Recall=%.4f  "
        "Precision=%.4f  TP=%d  FN=%d  FP=%d  (alerts=%d)",
        label, best_th,
        float(best_row["f2"]),    float(best_row["recall"]),
        float(best_row["precision"]),
        int(best_row["tp"]),      int(best_row["fn"]),
        int(best_row["fp"]),      int(best_row["predicted_pos"]),
    )
    return best_th, sweep


# Keep the old function name as an alias for backwards compatibility
# (other modules that import sweep_threshold will still work)
def sweep_threshold(
    y_true : np.ndarray,
    y_prob : np.ndarray,
    label  : str = "OOF",
) -> Tuple[float, pd.DataFrame]:
    """Alias for optimize_threshold_precision_first (backwards compatible)."""
    return optimize_threshold_precision_first(
        y_true, y_prob,
        min_precision=CFG["PRECISION_FLOOR"],
        label=label,
    )


# ── 9. Alert-budget operating points ─────────────────────────────────────────

def recommend_operating_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label : str = "OOF",
) -> pd.DataFrame:
    """
    Report metrics at fixed alert-budget percentiles (top 0.25% / 0.5% / 1%
    / 2% / 5% riskiest subscribers). This lets the business pick an operating
    point sized to actual campaign / call-centre capacity rather than an
    abstract probability cutoff.
    """
    rows = []
    n    = len(y_true)
    for pct in CFG["ALERT_BUDGET_PCTS"]:
        n_alert = max(1, int(n * pct))
        # Select the top-n_alert highest-scoring subscribers
        top_idx = np.argsort(y_prob)[-n_alert:]
        pred    = np.zeros(n, dtype=int)
        pred[top_idx] = 1

        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f2   = fbeta_score(y_true, pred, beta=2, zero_division=0)
        rows.append(dict(
            alert_pct   = f"{pct*100:.2f}%",
            n_alerts    = n_alert,
            precision   = round(prec, 4),
            recall      = round(rec,  4),
            f2          = round(f2,   4),
            tp=int(tp), fp=int(fp), fn=int(fn),
        ))

    df_op = pd.DataFrame(rows)
    log.info("[%s] Alert-budget operating points:\n%s", label, df_op.to_string(index=False))
    return df_op


# ── 10. Test-holdout evaluation ───────────────────────────────────────────────

def evaluate_on_test(
    train_df     : pd.DataFrame,
    test_df      : pd.DataFrame,
    selected_feat: List[str],
    best_th      : float,
    blend_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Re-train full ensemble on ALL of TRAIN, predict on TEST holdout.

    v3 changes:
      • SMOTE removed: full TRAIN used as-is with scale_pos_weight.
      • LR meta-learner removed: TEST predictions are rank-blended using
        the same AUC-proportional weights computed during OOF CV.
      • Returns `artifacts` dict that includes blend_weights for
        production SCORE-mode inference.

    Returns
    ───────
    test_prob : (n_test,) float32 — rank-blended score per TEST subscriber
    test_true : (n_test,) int    — ground truth
    test_pred : (n_test,) int    — binary predictions at best_th
    artifacts : dict             — fitted models + scaler + threshold + weights
    """
    log.info("Re-training full ensemble on entire TRAIN for TEST holdout …")
    X_tr_full = train_df[selected_feat].values.astype(float)
    y_tr_full = train_df[CFG["TARGET"]].values.astype(int)
    X_te      = test_df [selected_feat].values.astype(float)
    y_te      = test_df [CFG["TARGET"]].values.astype(int)

    # Scale
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr_full)
    X_te_s = scaler.transform(X_te)

    # Cost-sensitive scale_pos_weight from full training set
    neg_full = int((y_tr_full == 0).sum())
    pos_full = int((y_tr_full == 1).sum())
    spw_full = neg_full / max(pos_full, 1)
    log.info("Full-train scale_pos_weight = %.1f  (neg=%d, pos=%d)",
             spw_full, neg_full, pos_full)

    # Train three base models (no SMOTE)
    xgb_model = build_xgb(scale_pos_weight=spw_full)
    xgb_model.fit(X_tr_s, y_tr_full, verbose=False)

    rf_model = build_rf()
    rf_model.fit(X_tr_s, y_tr_full)

    et_model = build_et()
    et_model.fit(X_tr_s, y_tr_full)

    # Rank-blend TEST predictions
    p_xgb     = xgb_model.predict_proba(X_te_s)[:, 1]
    p_rf      = rf_model.predict_proba (X_te_s)[:, 1]
    p_et      = et_model.predict_proba (X_te_s)[:, 1]
    test_prob = rank_blend(p_xgb, p_rf, p_et, weights=blend_weights)
    test_pred = (test_prob >= best_th).astype(int)

    auc = roc_auc_score(y_te, test_prob) if len(np.unique(y_te)) > 1 else float("nan")
    pr  = average_precision_score(y_te, test_prob) if len(np.unique(y_te)) > 1 else float("nan")
    f2  = fbeta_score(y_te, test_pred, beta=2, zero_division=0)

    log.info(
        "TEST holdout → ROC-AUC=%.4f  PR-AUC=%.4f  F2=%.4f  "
        "Recall=%.4f  Precision=%.4f",
        auc, pr, f2,
        recall_score   (y_te, test_pred, zero_division=0),
        precision_score(y_te, test_pred, zero_division=0),
    )

    artifacts = {
        "scaler"        : scaler,
        "xgb"           : xgb_model,
        "rf"            : rf_model,
        "et"            : et_model,
        "threshold"     : best_th,
        "blend_weights" : blend_weights.tolist(),
        "selected_feat" : selected_feat,
        "scale_pos_weight": spw_full,
    }
    return test_prob, y_te, test_pred, artifacts


# ── 11. Model persistence ─────────────────────────────────────────────────────

def save_model_artifacts(artifacts: Dict, model_dir: Optional[str] = None) -> str:
    """Serialise trained artifacts to disk with joblib."""
    if not _JOBLIB_OK:
        log.warning("joblib unavailable — model not saved")
        return ""
    model_dir = model_dir or CFG["MODEL_DIR"]
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "churn_model_artifacts.joblib")
    joblib.dump(artifacts, path, compress=3)
    log.info("Model artifacts saved → %s", path)
    return path


def load_model_artifacts(model_dir: Optional[str] = None) -> Dict:
    """Load previously saved artifacts for SCORE mode."""
    if not _JOBLIB_OK:
        raise RuntimeError("joblib unavailable — cannot load model artifacts")
    model_dir = model_dir or CFG["MODEL_DIR"]
    path      = os.path.join(model_dir, "churn_model_artifacts.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifacts not found: {path}\n"
            f"Run in TRAIN mode first to generate them."
        )
    artifacts = joblib.load(path)
    log.info("Model artifacts loaded from %s", path)
    return artifacts


# ── 12. Batch scoring (SCORE mode) ────────────────────────────────────────────

def _score_batch(X_raw: np.ndarray, artifacts: Dict) -> np.ndarray:
    """
    Score a single batch of raw feature data.
    Uses rank blending with the blend_weights stored in artifacts.
    """
    scaler  = artifacts["scaler"]
    X_s     = scaler.transform(X_raw)
    p_xgb   = artifacts["xgb"].predict_proba(X_s)[:, 1]
    p_rf    = artifacts["rf"].predict_proba (X_s)[:, 1]
    p_et    = artifacts["et"].predict_proba (X_s)[:, 1]
    blended = rank_blend(p_xgb, p_rf, p_et,
                         weights=artifacts.get("blend_weights", None))
    return blended


def score_in_chunks(
    df          : pd.DataFrame,
    artifacts   : Dict,
    chunk_size  : int = 200_000,
) -> pd.DataFrame:
    """
    Score the full population in memory-safe chunks.
    Suitable for the 3.5M-row Oracle production table.
    """
    selected_feat = artifacts["selected_feat"]
    threshold     = artifacts["threshold"]
    id_col        = "MSISDN"

    # Ensure all feature columns are present (fill missing with 0)
    for c in selected_feat:
        if c not in df.columns:
            df[c] = 0.0

    # Winsorise and cast
    for col in selected_feat:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df[col].replace([np.inf, -np.inf], 0.0, inplace=True)

    n       = len(df)
    results = []
    for start in range(0, n, chunk_size):
        end    = min(start + chunk_size, n)
        chunk  = df.iloc[start:end]
        X_raw  = chunk[selected_feat].values.astype(CFG["SCORE_DTYPE"])
        probs  = _score_batch(X_raw, artifacts)
        out    = pd.DataFrame({
            id_col              : chunk[id_col].values if id_col in chunk.columns
                                   else np.arange(start, end),
            "churn_probability" : probs.astype(np.float32),
            "flagged"           : (probs >= threshold).astype(int),
        })
        results.append(out)
        log.info("  … scored rows %d–%d (%.1f%%)", start, end, end / n * 100)

    scored = pd.concat(results, ignore_index=True)
    scored["risk_decile"] = pd.qcut(
        scored["churn_probability"].rank(method="first"),
        10, labels=list(range(10, 0, -1))
    ).astype(int)

    log.info(
        "Scoring complete: %d rows | %d flagged (%.3f%%) at threshold=%.4f",
        len(scored), int(scored["flagged"].sum()),
        scored["flagged"].mean() * 100, threshold,
    )
    return scored


# ── 13. Reporting & Dashboard ─────────────────────────────────────────────────

def _sep(c="═", w=76): print(c * w)


def _print_cm(y_true, y_pred, label: str):
    cm          = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    pos         = tp + fn
    _sep("─", 60)
    print(f"  CONFUSION MATRIX  [{label}]")
    _sep("─", 60)
    print("                        Pred: Non-Churn    Pred: CHURN")
    print(f"  Actual: Non-Churn       {tn:>10,}       {fp:>10,}")
    print(f"  Actual: CHURN           {fn:>10,}       {tp:>10,}")
    _sep("─", 60)
    print(f"  Churners caught  (TPR) : {tp}/{pos}  = {tp/max(pos,1)*100:.1f}%  ✓")
    print(f"  Churners missed  (FNR) : {fn}/{pos}  = {fn/max(pos,1)*100:.1f}%  ✗")
    print(f"  False alarms     (FPR) : {fp:,}  (of {tn+fp:,} non-churners)")
    print(f"  Alert precision        : {tp/(max(tp+fp,1))*100:.1f}%")
    _sep("─", 60)
    print()


def full_report(
    oof_prob     : np.ndarray,
    oof_true     : np.ndarray,
    test_prob    : np.ndarray,
    test_true    : np.ndarray,
    test_pred    : np.ndarray,
    best_th      : float,
    fi_dict      : Dict,
    selected_feat: List[str],
    sweep_df     : pd.DataFrame,
):
    _sep()
    print("  CHURN PIPELINE v3 — EVALUATION REPORT  [Cost-Sensitive + Rank-Blend]")
    _sep()

    # OOF summary
    try:
        oof_auc = roc_auc_score(oof_true, oof_prob)
        oof_pr  = average_precision_score(oof_true, oof_prob)
        print(f"  OOF ROC-AUC : {oof_auc:.4f}")
        print(f"  OOF PR-AUC  : {oof_pr:.4f}")
    except Exception:
        pass

    # TEST summary
    if test_true is not None and len(test_true) > 0:
        try:
            test_auc = roc_auc_score(test_true, test_prob)
            test_pr  = average_precision_score(test_true, test_prob)
            print(f"  TEST ROC-AUC: {test_auc:.4f}")
            print(f"  TEST PR-AUC : {test_pr:.4f}")
        except Exception:
            pass
        print(f"  Threshold   : {best_th:.4f}  (F2-max, precision≥{CFG['PRECISION_FLOOR']:.2f})")
        print()
        _print_cm(test_true, test_pred, f"TEST holdout  [th={best_th:.4f}]")
        print(classification_report(test_true, test_pred,
                                    target_names=["No-Churn", "Churn"],
                                    zero_division=0))

    # Threshold sweep summary
    if sweep_df is not None and len(sweep_df):
        _sep("─", 60)
        print("  THRESHOLD SWEEP SUMMARY  (top 10 by F2, precision≥floor)")
        _sep("─", 60)
        floored = sweep_df[sweep_df["precision"] >= CFG["PRECISION_FLOOR"]]
        top10   = floored.nlargest(10, "f2") if len(floored) else sweep_df.head(10)
        print(top10[["threshold","precision","recall","f2","tp","fp","predicted_pos"]].to_string(index=False))
        print()

    # Feature importances
    _sep("─", 60)
    print("  TOP-20 FEATURE IMPORTANCES (XGBoost)")
    _sep("─", 60)
    fi_series = pd.Series(fi_dict["xgb"], index=selected_feat).sort_values(ascending=False)
    for feat, score in fi_series.head(20).items():
        tag = "  [W13-RATIO]" if "W13_RATIO" in feat else \
              "  [CONSEC-ZERO]" if "CONSEC_ZERO" in feat else ""
        print(f"  {feat:<45} {score:.5f}{tag}")
    print()
    _sep()


def _plot_dashboard(
    oof_prob  : np.ndarray,
    oof_true  : np.ndarray,
    test_prob : np.ndarray,
    test_true : np.ndarray,
    test_pred : np.ndarray,
    best_th   : float,
    fi_dict   : Dict,
    feat_names: List[str],
    sweep_df  : pd.DataFrame,
):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle("Churn Pipeline v3 — Cost-Sensitive + Rank-Blend Dashboard",
                 fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.36)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # ── Panel 1: OOF ROC curve ────────────────────────────────────────────────
    try:
        fpr, tpr, _ = roc_curve(oof_true, oof_prob)
        auc_val     = roc_auc_score(oof_true, oof_prob)
        ax1.plot(fpr, tpr, color="#2980B9", lw=2, label=f"OOF AUC={auc_val:.4f}")
        ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
        ax1.set_title("OOF ROC Curve", fontweight="bold")
        ax1.legend(fontsize=9); ax1.spines[["top","right"]].set_visible(False)
    except Exception:
        ax1.set_visible(False)

    # ── Panel 2: OOF Precision-Recall curve ───────────────────────────────────
    try:
        prec_c, rec_c, _ = precision_recall_curve(oof_true, oof_prob)
        pr_auc = average_precision_score(oof_true, oof_prob)
        ax2.plot(rec_c, prec_c, color="#C0392B", lw=2, label=f"OOF PR-AUC={pr_auc:.4f}")
        base = oof_true.mean()
        ax2.axhline(base, color="gray", ls="--", lw=1, label=f"Base rate ({base*100:.2f}%)")
        ax2.axhline(CFG["PRECISION_FLOOR"], color="black", ls="-.", lw=1.2,
                    label=f"Precision floor ({CFG['PRECISION_FLOOR']*100:.0f}%)")
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.set_title("OOF Precision-Recall Curve", fontweight="bold")
        ax2.legend(fontsize=9); ax2.spines[["top","right"]].set_visible(False)
    except Exception:
        ax2.set_visible(False)

    # ── Panel 3: Threshold sweep (F2 and Precision vs threshold) ─────────────
    if sweep_df is not None and len(sweep_df):
        ax3b = ax3.twinx()
        ax3.plot(sweep_df["threshold"], sweep_df["f2"],
                 color="#2980B9", lw=2, label="F2 score")
        ax3b.plot(sweep_df["threshold"], sweep_df["precision"],
                  color="#E67E22", lw=1.5, ls="--", label="Precision")
        ax3.axvline(best_th, color="black", ls="-.", lw=1.5, label=f"Selected th={best_th:.3f}")
        ax3.axhline(0, color="gray", ls=":", lw=0.8)
        ax3b.axhline(CFG["PRECISION_FLOOR"], color="#E67E22", ls=":", lw=0.8,
                     label=f"Floor {CFG['PRECISION_FLOOR']:.2f}")
        ax3.set_xlabel("Threshold"); ax3.set_ylabel("F2 Score", color="#2980B9")
        ax3b.set_ylabel("Precision", color="#E67E22")
        ax3.set_title("Threshold Sweep: F2 & Precision", fontweight="bold")
        lines1, labs1 = ax3.get_legend_handles_labels()
        lines2, labs2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
        ax3.spines[["top"]].set_visible(False)

    # ── Panel 4: OOF probability distribution by class ────────────────────────
    try:
        bins = np.linspace(0, 1, 50)
        ax4.hist(oof_prob[oof_true == 0], bins=bins, alpha=0.6, density=True,
                 color="#27AE60", label="Non-Churn")
        ax4.hist(oof_prob[oof_true == 1], bins=bins, alpha=0.8, density=True,
                 color="#C0392B", label="Churn")
        ax4.axvline(best_th, color="black", ls="-.", lw=1.5,
                    label=f"Threshold={best_th:.3f}")
        ax4.set_title("OOF Rank-Blend Score Distribution", fontweight="bold")
        ax4.set_xlabel("Rank-Blend Score"); ax4.set_ylabel("Density")
        ax4.legend(fontsize=9); ax4.spines[["top","right"]].set_visible(False)
    except Exception:
        ax4.set_visible(False)

    # ── Panel 5: TEST confusion matrix heatmap ────────────────────────────────
    if test_true is not None and len(test_true) > 0:
        try:
            cm   = confusion_matrix(test_true, test_pred)
            tn_v, fp_v, fn_v, tp_v = cm.ravel()
            ann  = np.array([[f"TN\n{tn_v:,}", f"FP\n{fp_v:,}"],
                              [f"FN\n{fn_v:,}", f"TP\n{tp_v:,}"]])
            sns.heatmap(cm, annot=ann, fmt="", cmap="Reds", linewidths=2,
                        linecolor="white", ax=ax5, cbar=False,
                        xticklabels=["Pred: No-Churn", "Pred: Churn"],
                        yticklabels=["Actual: No-Churn", "Actual: Churn"],
                        annot_kws={"size": 11, "weight": "bold"})
            prec_v = tp_v / max(tp_v + fp_v, 1)
            rec_v  = tp_v / max(tp_v + fn_v, 1)
            ax5.set_title(
                f"TEST Confusion Matrix\n"
                f"Precision={prec_v*100:.1f}%  Recall={rec_v*100:.1f}%",
                fontweight="bold")
            ax5.tick_params(labelsize=9)
        except Exception:
            ax5.set_visible(False)

    # ── Panel 6: Feature importance (XGBoost top-20) ─────────────────────────
    try:
        fi_series = pd.Series(fi_dict["xgb"], index=feat_names).sort_values()
        top20     = fi_series.tail(20)
        colors    = ["#E67E22" if ("W13_RATIO" in f or "CONSEC_ZERO" in f)
                     else "#2980B9" for f in top20.index]
        ax6.barh(range(len(top20)), top20.values, color=colors, edgecolor="white")
        ax6.set_yticks(range(len(top20)))
        ax6.set_yticklabels(
            [f.replace("_", " ").title()[:30] for f in top20.index], fontsize=7.5
        )
        ax6.set_title("Top-20 Feature Importance (XGBoost)\n"
                      "[orange = v3 W13-ratio / consec-zero features]",
                      fontweight="bold", fontsize=9)
        ax6.set_xlabel("Importance score")
        ax6.spines[["top", "right"]].set_visible(False)
    except Exception:
        ax6.set_visible(False)

    path = os.path.join(CFG["OUTPUT_DIR"], "churn_v2_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Dashboard saved → %s", path)


# ── 14. Raw loader for SCORE mode ─────────────────────────────────────────────

def _load_raw_for_scoring() -> pd.DataFrame:
    """Load and engineer features for SCORE-mode inference (no label required)."""
    mode = CFG["RUN_MODE"]
    if mode == "ORACLE":
        df = _fetch_oracle()
    else:
        df = pd.read_csv(CFG["INPUT_CSV"])
        df.columns = [c.upper().strip() for c in df.columns]

    if "AON" in df.columns:
        df["AON"] = pd.to_numeric(df["AON"], errors="coerce").fillna(0)
    df = engineer_extra_features(df)
    return df


# ── 15. Training pipeline ─────────────────────────────────────────────────────

def run_training_pipeline() -> None:
    t0 = time.time()
    _sep()
    print("  CHURN PIPELINE v3  |  Mode: TRAIN  |  Source:", CFG["RUN_MODE"])
    print("  Cost-sensitive learning + Rank-blend ensemble")
    print("  Precision-first threshold (floor=", CFG["PRECISION_FLOOR"], ")")
    _sep()

    # ── Load & clean ──────────────────────────────────────────────────────────
    train_df, test_df, feat_cols = load_and_clean()
    log.info("TRAIN: %d rows | TEST: %d rows | Features: %d",
             len(train_df), len(test_df), len(feat_cols))

    # ── Feature selection ─────────────────────────────────────────────────────
    selected_feat, mi_scores = select_features(
        train_df, feat_cols, top_k=CFG["TOP_K_FEATURES"]
    )

    # Save MI scores for audit
    mi_df = pd.Series(mi_scores, index=feat_cols).sort_values(ascending=False)
    mi_df.to_csv(os.path.join(CFG["OUTPUT_DIR"], "feature_importance.csv"),
                 header=["mi_score"])

    # ── 5-Fold CV (rank-blend, no SMOTE) ─────────────────────────────────────
    oof_blend, oof_true, fi_dict, blend_weights = run_stacked_cv(
        train_df, selected_feat, feat_cols
    )

    # ── OOF threshold optimisation (precision-first) ──────────────────────────
    best_th, sweep_df = optimize_threshold_precision_first(
        oof_true, oof_blend,
        min_precision=CFG["PRECISION_FLOOR"],
        label="OOF",
    )
    sweep_df.to_csv(
        os.path.join(CFG["OUTPUT_DIR"], "oof_threshold_sweep.csv"), index=False
    )
    log.info("OOF threshold sweep saved.")

    # OOF alert-budget operating points
    recommend_operating_points(oof_true, oof_blend, label="OOF")

    # ── TEST evaluation ───────────────────────────────────────────────────────
    test_prob, test_true, test_pred, artifacts = evaluate_on_test(
        train_df, test_df, selected_feat, best_th, blend_weights
    )

    # TEST threshold sweep (diagnostic only — do not re-select threshold)
    _, test_sweep_df = optimize_threshold_precision_first(
        test_true, test_prob,
        min_precision=CFG["PRECISION_FLOOR"],
        label="TEST-diagnostic",
    )
    test_sweep_df.to_csv(
        os.path.join(CFG["OUTPUT_DIR"], "test_threshold_sweep.csv"), index=False
    )

    # TEST alert-budget operating points
    recommend_operating_points(test_true, test_prob, label="TEST")

    # ── Report ────────────────────────────────────────────────────────────────
    full_report(
        oof_prob=oof_blend, oof_true=oof_true,
        test_prob=test_prob, test_true=test_true, test_pred=test_pred,
        best_th=best_th, fi_dict=fi_dict,
        selected_feat=selected_feat, sweep_df=sweep_df,
    )

    # ── Dashboard ─────────────────────────────────────────────────────────────
    _plot_dashboard(
        oof_prob=oof_blend, oof_true=oof_true,
        test_prob=test_prob, test_true=test_true, test_pred=test_pred,
        best_th=best_th, fi_dict=fi_dict,
        feat_names=selected_feat, sweep_df=sweep_df,
    )

    # ── Save model artifacts ──────────────────────────────────────────────────
    save_model_artifacts(artifacts)

    _sep()
    print(f"  Pipeline complete in {time.time()-t0:.1f}s.  Outputs: {CFG['OUTPUT_DIR']}")
    print(f"  Model saved → {CFG['MODEL_DIR']}/churn_model_artifacts.joblib")
    print(f"  To score a large population:")
    print(f"      MODE=SCORE RUN_MODE=ORACLE python churn_pipeline_v2.py")
    print(f"      MODE=SCORE RUN_MODE=CSV INPUT_CSV=<file> python churn_pipeline_v2.py")
    _sep()


# ── 16. Scoring pipeline (SCORE mode) ────────────────────────────────────────

def run_scoring_pipeline() -> None:
    t0 = time.time()
    _sep()
    print(f"  CHURN PIPELINE v3  —  Mode: SCORE  |  Source: {CFG['RUN_MODE']}")
    _sep()

    artifacts = load_model_artifacts()
    df        = _load_raw_for_scoring()
    scored    = score_in_chunks(df, artifacts, chunk_size=CFG["SCORE_CHUNK_SIZE"])

    out_path = os.path.join(CFG["OUTPUT_DIR"], "production_churn_scores.csv")
    scored.to_csv(out_path, index=False)

    n_flagged = int(scored["flagged"].sum())
    _sep()
    print(f"  SCORING COMPLETE")
    print(f"  Rows scored      : {len(scored):,}")
    print(f"  Flagged at risk  : {n_flagged:,}  ({n_flagged/len(scored)*100:.3f}%)")
    print(f"  Threshold used   : {artifacts['threshold']:.4f}")
    print(f"  Blend weights    : XGB={artifacts['blend_weights'][0]:.3f}  "
          f"RF={artifacts['blend_weights'][1]:.3f}  "
          f"ET={artifacts['blend_weights'][2]:.3f}")
    print(f"  Output           : {out_path}")
    print(f"  Elapsed          : {time.time()-t0:.1f}s")
    _sep()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    mode = os.getenv("MODE", "TRAIN").upper()
    if mode == "SCORE":
        run_scoring_pipeline()
    else:
        run_training_pipeline()


if __name__ == "__main__":
    main()
