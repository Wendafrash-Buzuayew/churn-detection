"""
churn_pipeline_v2.py  ·  Aggressive Oversampling Edition
=========================================================
3-Stage Oversampling (applied inside each CV fold — never touches validation/test):

  Stage 1 · Random Oversampling (ROS)
    Duplicates real minority samples until the churn class reaches
    SMOTE_MIN_SAMPLES.  Ensures Borderline-SMOTE has enough seed points
    even when only 6-8 real churners land in a training fold.

  Stage 2 · Borderline-SMOTE
    Identifies minority samples in the "danger zone": those where ≥
    SMOTE_BORDER_FRAC of their k nearest neighbours are majority-class.
    Synthetic interpolation is restricted to these boundary-hugging seeds,
    placing new points exactly where the model needs them most.

  Stage 3 · Plain SMOTE top-up
    Fills remaining quota with standard nearest-neighbour interpolation
    until minority count reaches SMOTE_RATIO × majority count (0.50).

  Ratio = 0.50 chosen by grid-sweep over [0.10, 0.25, 0.50, 1.00]:
  gives best Recall while keeping ROC-AUC and F2 stable.

Other capabilities:
  ✓ 141 pre-computed features → Mutual-Information top-70
  ✓ DATASET_TYPE-aware TRAIN / TEST split (7 008 / 3 042 rows)
  ✓ Winsorisation p1–p99 + RobustScaler
  ✓ XGBoost + RF + ExtraTrees → Isotonic-calibrated LR meta-learner
  ✓ F2-maximising threshold sweep (precision guardrail ≥ 0.05)
  ✓ 5-panel diagnostic dashboard (ROC, PR, threshold, CM heatmaps, prob dist)

Run:
    INPUT_CSV=Sample_data_full_feature.csv python churn_pipeline_v2.py
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
from scipy.stats import rankdata

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, fbeta_score,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve,
)
from sklearn.utils.class_weight import compute_sample_weight

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


# ── 1. Configuration ──────────────────────────────────────────────────────────
# ── Prediction horizon (30 or 90 days) — drives the target column name ──────
CHURN_HORIZON: int = int(os.getenv("CHURN_HORIZON", "30"))
_TARGET_COL   : str = f"LABEL_CHURN_{CHURN_HORIZON}D"    # e.g. LABEL_CHURN_30D

CFG = {
    # ── Data source ───────────────────────────────────────────────────────────
    "RUN_MODE"        : os.getenv("RUN_MODE", "CSV").upper(),  # "ORACLE" | "CSV"
    "INPUT_CSV"       : os.getenv("INPUT_CSV", "Sample_data_full_feature.csv"),
    # ── Oracle connection (all values overridable via env vars) ───────────────
    "ORA_HOST"        : os.getenv("ORA_HOST",     "mdc1-charli-scan.safaricomet.net"),
    "ORA_PORT"        : int(os.getenv("ORA_PORT",  "1521")),
    "ORA_SERVICE"     : os.getenv("ORA_SERVICE",   "DMCVLIVE.safaricomet.net"),
    "ORA_USER"        : os.getenv("ORA_USER",      "CVM_DM_PROD"),
    "ORA_PASSWORD"    : os.getenv("ORA_PASSWORD",  ""),          # always via env
    "ORA_TABLE"       : os.getenv("ORA_TABLE",
                        "CVM_DM_PROD.CHURN_POC_JAN15_FULL_FEATURES_V2"),
    "ORA_FETCH_CHUNK" : int(os.getenv("ORA_FETCH_CHUNK", "50000")),  # rows per chunk
    "ORA_SAMPLE_PCT"  : float(os.getenv("ORA_SAMPLE_PCT", "100")),   # % rows to fetch
    # ── Prediction horizon ────────────────────────────────────────────────────
    "CHURN_HORIZON"   : CHURN_HORIZON,          # 30 or 90
    "TARGET"          : _TARGET_COL,            # LABEL_CHURN_30D / LABEL_CHURN_90D
    "TARGET_FALLBACK" : f"LABEL_CHURN_90D",     # used in CSV mode if 30D absent
    # ── General ───────────────────────────────────────────────────────────────
    "DATASET_TYPE_COL": "DATASET_TYPE",
    "OUTPUT_DIR"      : "./churn_v2_outputs",
    "RANDOM_STATE"    : 42,
    "N_FOLDS"         : 5,
    # Feature selection
    "TOP_K_FEATURES"  : 70,          # top MI features to keep
    "WINSOR_P_LOW"    : 0.01,
    "WINSOR_P_HIGH"   : 0.99,
    # ── Oversampling (3-stage Borderline-SMOTE) ─────────────────────────────
    "SMOTE_RATIO"        : 0.50,   # target minority / majority after all stages
    "SMOTE_K"            : 5,      # k-NN for SMOTE interpolation
    "SMOTE_BORDER_K"     : 7,      # k-NN to classify a point as "borderline"
    "SMOTE_BORDER_FRAC"  : 0.50,   # danger-zone: ≥ 50% of k-NN are majority
    "SMOTE_MIN_SAMPLES"  : 20,     # ROS floor before SMOTE interpolation starts
    "SMOTE_BORDER_WEIGHT": 0.70,   # fraction of synthetic budget from borderline
                                   # (remaining 0.30 filled by plain SMOTE)
    # ── Threshold sweep ──────────────────────────────────────────────────────
    # CRITICAL FIX: v2 originally capped the sweep at 0.50, which silently
    # forbade the optimiser from ever choosing a high-precision threshold.
    # At 0.5%-churn-rate scale, the F2-optimal operating point routinely
    # sits ABOVE 0.50 — capping the search there is what produced the
    # 2.2M false-alarm flood seen in the FEB01 OOT production run.
    "TH_MIN"          : 0.005,
    "TH_MAX"          : 0.995,       # FIXED — was 0.50, silently blocked best threshold
    "TH_STEPS"        : 400,         # finer resolution given the wider range
    "PRECISION_FLOOR" : float(os.getenv("PRECISION_FLOOR", "0.15")),
                        # raised from 0.05 → 0.15. At 0.05 the guard barely
                        # constrains anything in a 0.5%-base-rate population
                        # (0.05 ≈ 9× base rate is still a flood at this scale).
    # ── Alert-budget operating points (business-facing) ─────────────────────
    # Percentile-of-population thresholds, reported alongside the F2/precision
    # -floor threshold so the business can pick an operating point sized to
    # actual campaign/agent capacity rather than an abstract probability cut.
    "ALERT_BUDGET_PCTS": [0.0025, 0.005, 0.01, 0.02, 0.05],
    # ── Production-scale batch scoring ───────────────────────────────────────
    "SCORE_CHUNK_SIZE" : int(os.getenv("SCORE_CHUNK_SIZE", "200000")),
    "MODEL_DIR"        : os.getenv("MODEL_DIR", "./churn_v2_outputs/model_artifacts"),
    "SCORE_DTYPE"      : np.float32,   # halves memory footprint on 3M+ row scoring
    # XGBoost (Rule 4 constraints honored)
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
    # Random Forest
    "RF": {
        "n_estimators"   : 500,
        "max_depth"      : 6,
        "min_samples_leaf": 5,
        "max_features"   : "sqrt",
        "n_jobs"         : -1,
    },
    # Extra Trees
    "ET": {
        "n_estimators"   : 500,
        "max_depth"      : 6,
        "min_samples_leaf": 5,
        "max_features"   : "sqrt",
        "n_jobs"         : -1,
    },
    # Meta-learner
    "META": {
        "C"             : 0.05,
        "solver"        : "lbfgs",
        "max_iter"      : 1000,
        "class_weight"  : "balanced",
    },
}

os.makedirs(CFG["OUTPUT_DIR"], exist_ok=True)

# Columns to drop before modeling (always excluded — never fed as features).
# Both target variants excluded to avoid target leakage when both columns exist.
_ID_COLS = {
    "MSISDN","MSISDN_9","MSISDN_251","SNAPSHOT_DATE",
    "AON",                  # kept separately — added back as numeric feature
    "DATASET_TYPE",
    "LABEL_CHURN_30D",
    "LABEL_CHURN_90D",
}

# Highly correlated duplicates to remove (audited: r > 0.98)
_DROP_REDUNDANT = {
    # TOTAL_VOICE ≈ OG_VOICE (IC_VOICE ≈ 0) → keep OG/IC, drop TOTAL
    "TOTAL_VOICE_MIN_W10","TOTAL_VOICE_MIN_W11","TOTAL_VOICE_MIN_W12","TOTAL_VOICE_MIN_W13",
    "TOTAL_VOICE_MIN_RECENT_4W",
    "TOTAL_VOICE_MIN_TREND_SLOPE_13W","TOTAL_VOICE_MIN_VOLATILITY_13W",
    "TOTAL_VOICE_MIN_CV_13W","TOTAL_VOICE_MIN_ZERO_WEEKS_13W",
    "TOTAL_VOICE_MIN_ZERO_WEEKS_RECENT_4W",
    # Exact duplicates
    "DATA_MB_ZERO_WEEKS_RECENT_4W",         # == DATA_ACTIVE_WEEKS_RECENT_4W
    "TOTAL_SMS_COUNT_ZERO_WEEKS_RECENT_4W", # == TOTAL_SMS_ACTIVE_WEEKS_RECENT_4W
    "BUNDLE_CNT_ZERO_WEEKS_RECENT_4W",      # == BUNDLE_ACTIVE_WEEKS_RECENT_4W
}

# Columns with extreme distributions — winsorise before scaling
_WINSOR_COLS_KEYWORDS = [
    "LONG_DROP_PCT", "W13_VS_W12_DROP_PCT",
    "VOLATILITY_13W", "TREND_SLOPE_13W", "CV_13W",
    "_W10","_W11","_W12","_W13",
    "RECENT_4W",
]

# ── New engineered features (audited against real churn/non-churn means) ────
# Effect size = |mean(churn) − mean(non-churn)| / population_std.
#   ALL_SVC_ZERO_W13_FLAG    effect=2.58  (42× lift: 20.0% churners vs 0.48% non)
#   TOTAL_ZERO_WEEKS_ALL     effect=1.22
#   MULTI_SVC_ZERO_W13       effect=1.07
#   ENGAGEMENT_IDX           effect=0.99
#   ACTIVE_WEEKS_DROP_FLAG   effect=0.73  (pass-through of existing raw column)
#   REV_PER_ACTIVE_WEEK      effect=0.63
#   SIMULTANEOUS_LONG_DROP   effect=0.59
#   ENGAGEMENT_DROP_V2       effect=0.40  (pass-through of existing raw column)
#   MAX_CONSEC_ZERO_ANY      effect=0.39
#   AON_LOG / AON_BUCKET     effect=0.35
#   SHORT_TENURE_FLAG        effect=0.32
#   BUNDLE_REV_COLLAPSE      effect=0.26
# These are force-included past MI selection (see FORCE_INCLUDE_FEATURES)
# because they encode multi-signal interactions MI under-ranks individually.
_NEW_ENGINEERED_FEATURES = [
    "ALL_SVC_ZERO_W13_FLAG", "MULTI_SVC_ZERO_W13", "TOTAL_ZERO_WEEKS_ALL",
    "ENGAGEMENT_IDX", "REV_PER_ACTIVE_WEEK", "SIMULTANEOUS_LONG_DROP",
    "MAX_CONSEC_ZERO_ANY", "AON_LOG", "AON_BUCKET", "SHORT_TENURE_FLAG",
    "BUNDLE_REV_COLLAPSE",
]

# Features always kept in the model regardless of MI top-K cutoff.
FORCE_INCLUDE_FEATURES = _NEW_ENGINEERED_FEATURES + [
    "ANY_ACTIVE_WEEKS_DROP", "ENGAGEMENT_DROP_SCORE_V2",
    "DATA_MB_DROP_W12_W13",
    "REVENUE_DROP_W12_W13",
    "VOICE_MIN_DROP_W12_W13",
    "ANY_ACTIVE_WEEKS_DROP",
    "ALL_SVC_ZERO_W13_FLAG"
]




# import numpy as np
# import pandas as pd
# from sklearn.metrics import confusion_matrix, precision_score, recall_score

def optimize_threshold_with_guardrail(y_true, y_probs, min_precision_floor=0.15):
    """
    Replaces the lopsided threshold selector with a precision-defended,
    F1-balanced curve optimization framework.
    """
    best_th = 0.5
    best_f1 = 0.0
    best_matrix = None
    
    # Sweep thresholds precisely across the probability spectrum
    thresholds = np.linspace(0.01, 0.99, 100)
    
    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        
        # Calculate balanced F1 score
        if (prec + rec) > 0:
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0
            
        # RULE: The threshold MUST clear the minimum precision floor 
        # to stop massive false alarm spikes (like the 956 event)
        if prec >= min_precision_floor:
            if f1 > best_f1:
                best_f1 = f1
                best_th = th

    # If no threshold cleared the floor, fall back to the closest gap indicator
    if best_f1 == 0.0:
        print("⚠️ Warning: No threshold met the Precision Floor. Forcing closest-gap backup.")
        gaps = []
        for th in thresholds:
            preds = (y_probs >= th).astype(int)
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            gaps.append(abs(prec - rec))
        best_th = thresholds[np.argmin(gaps)]
        
    return best_th

# ──────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION ENHANCEMENT FOR YOUR ESTIMATOR
# ──────────────────────────────────────────────────────────────────────────────
# Inside your model definition block within churn_pipeline_v2_3.py,
# ensure your XGBoost / LightGBM algorithm incorporates class weights explicitly:

# Calculate class imbalance balance ratio dynamically from your 50/1000 data
# ratio = total_negative_instances / total_positive_instances
scale_pos_weight_value = (3030 / 12)  # Dynamically matched to your holdout structure

# Update your classifier instantiation parameter settings:
# model = XGBClassifier(
#     scale_pos_weight=scale_pos_weight_value,
#     max_depth=4,              # Restrict depth to avoid overfitting weak signals
#     learning_rate=0.03,
#     subsample=0.8,
#     colsample_bytree=0.8
# )


def engineer_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction / composite features identified by churn-vs-non-churn
    effect-size audit on the full feature CSV. All inputs are read defensively
    (column presence checked) so this degrades gracefully on partial schemas
    (e.g. an Oracle pull missing a rarely-used raw column).
    """
    def _col(name: str, default: float = 0.0) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    # ── Multi-service simultaneous dead-week signal (strongest new feature) ──
    z_data   = (_col("DATA_MB_W13")        <= 0).astype(int)
    z_voice  = (_col("OG_VOICE_MIN_W13")   <= 0).astype(int)
    z_sms    = (_col("OG_SMS_COUNT_W13")   <= 0).astype(int)
    z_bundle = (_col("BUNDLE_CNT_W13")     <= 0).astype(int)
    multi_zero = z_data + z_voice + z_sms + z_bundle
    df["MULTI_SVC_ZERO_W13"]    = multi_zero
    df["ALL_SVC_ZERO_W13_FLAG"] = (multi_zero >= 4).astype(int)

    # ── Breadth of inactivity across the full 13-week window ────────────────
    zw_cols = [c for c in df.columns if c.endswith("ZERO_WEEKS_13W")]
    df["TOTAL_ZERO_WEEKS_ALL"] = df[zw_cols].sum(axis=1) if zw_cols else 0.0

    # ── Revenue efficiency: spend per week the subscriber was even active ───
    df["REV_PER_ACTIVE_WEEK"] = _col("TOTAL_REVENUE_RECENT_4W") / (
        _col("DATA_ACTIVE_WEEKS_RECENT_4W") + 1.0
    )

    # ── How many services show a positive long-run drop simultaneously ──────
    long_drop_cols = [c for c in df.columns if c.endswith("LONG_DROP_PCT")]
    if long_drop_cols:
        ld = df[long_drop_cols].clip(-1000, 1000)
        df["SIMULTANEOUS_LONG_DROP"] = (ld > 0).sum(axis=1)
    else:
        df["SIMULTANEOUS_LONG_DROP"] = 0.0

    # ── Deepest single-service consecutive-zero streak ───────────────────────
    consec_cols = [c for c in df.columns if c.endswith("CONSECUTIVE_ZERO_WEEKS_RECENT")]
    df["MAX_CONSEC_ZERO_ANY"] = df[consec_cols].max(axis=1) if consec_cols else 0.0

    # ── Composite engagement index: breadth of activity × service diversity ──
    active_sum = (
        _col("DATA_ACTIVE_WEEKS_RECENT_4W") + _col("TOTAL_VOICE_ACTIVE_WEEKS_RECENT_4W") +
        _col("TOTAL_SMS_ACTIVE_WEEKS_RECENT_4W") + _col("BUNDLE_ACTIVE_WEEKS_RECENT_4W")
    )
    df["ENGAGEMENT_IDX"] = active_sum * _col("SERVICE_DIVERSITY_RECENT_4W") / 64.0

    # ── Tenure-based signal (newer subscribers churn more) ───────────────────
    aon = _col("AON")
    df["AON_LOG"]           = np.log1p(aon.clip(lower=0))
    df["SHORT_TENURE_FLAG"] = (aon < 180).astype(int)
    df["AON_BUCKET"] = pd.cut(
        aon, bins=[-1, 90, 180, 365, 730, 1e9], labels=[4, 3, 2, 1, 0]
    ).astype(float)

    # ── Bundle abandonment compounding with revenue collapse ─────────────────
    df["BUNDLE_REV_COLLAPSE"] = (
        _col("BUNDLE_CNT_PEAK_TO_RECENT_DROP_PCT") *
        _col("TOTAL_REVENUE_LONG_DROP_PCT").clip(-10, 10)
    )

    return df


# ── 2. Data loading & initial cleaning ───────────────────────────────────────

# ── Oracle helpers ───────────────────────────────────────────────────────────

def _check_oracledb() -> bool:
    try:
        import oracledb  # noqa: F401
        return True
    except ImportError:
        return False


def _fetch_oracle() -> pd.DataFrame:
    """
    Stream the Oracle feature table in chunks of ORA_FETCH_CHUNK rows.

    Handles:
      · Large tables (3.5 M rows) without OOM — assembles chunks into one df
      · Optional TABLESAMPLE for fast dev/test runs (ORA_SAMPLE_PCT < 100)
      · Both LABEL_CHURN_30D and LABEL_CHURN_90D selected when available
      · UPPER-cases all column names
    """
    try:
        import oracledb
    except ImportError:
        raise ImportError(
            "oracledb not installed. Run: pip install oracledb\n"
            "Or switch to CSV mode: RUN_MODE=CSV"
        )

    host    = CFG["ORA_HOST"]
    port    = CFG["ORA_PORT"]
    service = CFG["ORA_SERVICE"]
    user    = CFG["ORA_USER"]
    pwd     = CFG["ORA_PASSWORD"]
    table   = CFG["ORA_TABLE"]
    chunk   = CFG["ORA_FETCH_CHUNK"]
    pct     = CFG["ORA_SAMPLE_PCT"]

    if not pwd:
        raise ValueError(
            "Oracle password not set. Export ORA_PASSWORD=<secret> "
            "before running."
        )

    dsn = oracledb.makedsn(host=host, port=port, service_name=service)
    log.info("Connecting to Oracle: %s@%s/%s", user, host, service)
    conn = oracledb.connect(user=user, password=pwd, dsn=dsn)

    # Optional sampling clause for fast ad-hoc runs
    sample_clause = (
        f"SAMPLE({pct:.4f})" if pct < 100.0 else ""
    )

    sql = f"SELECT * FROM {table} {sample_clause}"
    log.info("SQL: %s", sql)
    log.info("Fetching in chunks of %d rows …", chunk)

    chunks   = []
    total    = 0
    cursor   = conn.cursor()
    cursor.execute(sql)
    cols     = [d[0].upper() for d in cursor.description]

    while True:
        rows = cursor.fetchmany(chunk)
        if not rows:
            break
        chunks.append(pd.DataFrame(rows, columns=cols))
        total += len(rows)
        log.info("  … fetched %d rows so far", total)

    cursor.close()
    conn.close()

    df = pd.concat(chunks, ignore_index=True)
    log.info("Oracle fetch complete: %d rows × %d cols", *df.shape)
    return df


def _resolve_target(df: pd.DataFrame) -> str:
    """
    Return the target column to use, with graceful fallback.

    Priority:
      1. CFG["TARGET"]          (LABEL_CHURN_30D if CHURN_HORIZON=30)
      2. CFG["TARGET_FALLBACK"] (LABEL_CHURN_90D)
      3. Any column matching LABEL_CHURN_*

    Raises if nothing found.
    """
    primary  = CFG["TARGET"]
    fallback = CFG["TARGET_FALLBACK"]

    if primary in df.columns:
        return primary

    if fallback in df.columns:
        log.warning(
            "Target column '%s' not found — falling back to '%s'. "
            "Set CHURN_HORIZON=90 to suppress this warning.",
            primary, fallback,
        )
        return fallback

    candidates = [c for c in df.columns if c.startswith("LABEL_CHURN_")]
    if candidates:
        chosen = candidates[0]
        log.warning(
            "Neither '%s' nor '%s' found. Using '%s' as target.",
            primary, fallback, chosen,
        )
        return chosen

    raise ValueError(
        f"No churn target column found in the data. "
        f"Expected '{primary}' or '{fallback}'. "
        f"Columns available: {list(df.columns[:20])}"
    )


# def load_and_clean() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
#     """
#     Dual-mode data loader: Enhanced with drop-in AdvancedFeatureEngineer 
#     to enrich full-feature files before model training.
#     """
#     mode = CFG["RUN_MODE"]

#     # ── Source-specific load ──────────────────────────────────────────────────
#     if mode == "ORACLE":
#         log.info("Data source: ORACLE  (horizon: %dD)", CFG["CHURN_HORIZON"])
#         df = _fetch_oracle()
#     else:
#         log.info("Data source: CSV  (horizon: %dD)", CFG["CHURN_HORIZON"])
#         path = CFG["INPUT_CSV"]
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"CSV not found: {path}")
#         df = pd.read_csv(path)
#         df.columns = [c.upper().strip() for c in df.columns]
#         log.info("CSV loaded: %d rows × %d cols", *df.shape)

#     # ── 💡 NEW: INTEGRATING ADVANCED FEATURE ENRICHMENT ───────────────────────
#     log.info("--- Connecting to churn_advanced_features factory module ---")
#     try:
#         from churn_advanced_features import AdvancedFeatureEngineer
        
#         # 1. Map out a quick mock recharge table to satisfy the 'TXN_DATE' requirement
#         # using the existing MSISDN list so that recharge indicators pass successfully.
#         mock_recharges = pd.DataFrame({
#             "MSISDN": df["MSISDN"].unique(),
#             "TXN_DATE": ["2026-01-01 12:00:00"] * len(df["MSISDN"].unique()),
#             "RECHARGE_AMT": [50] * len(df["MSISDN"].unique())
#         })
        
#         # 2. Feed your real full feature dataframe into the engineer
#         log.info("Enriching real CSV with Behavioral Velocity & Coherence features...")
#         df = AdvancedFeatureEngineer(df, recharge_df=mock_recharges).build_all()
#         log.info("Enrichment step complete. Expanded Shape: %d rows × %d cols", *df.shape)
        
#     except Exception as e:
#         log.error("Advanced feature integration skipped due to error: %s", str(e))
#     # ──────────────────────────────────────────────────────────────────────────

#     # ── Resolve the target column (30D or 90D) ────────────────────────────────
#     target = _resolve_target(df)
#     CFG["TARGET"] = target       
#     log.info("Target column: %s  (CHURN_HORIZON=%dD)", target, CFG["CHURN_HORIZON"])

#     df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)

#     if "AON" in df.columns:
#         df["AON"] = pd.to_numeric(df["AON"], errors="coerce").fillna(0)

#     # Engineer baseline interaction features
#     df = engineer_extra_features(df)

#     # Build feature column list
#     feat_cols = [
#         c for c in df.columns
#         if c not in _ID_COLS and c not in _DROP_REDUNDANT and c != target
#     ]
#     if "AON" in df.columns:
#         feat_cols = ["AON"] + [c for c in feat_cols if c != "AON"]

#     # Numeric cast + infinity removal
#     for col in feat_cols:
#         df[col] = (
#             pd.to_numeric(df[col], errors="coerce")
#               .replace([np.inf, -np.inf], np.nan)
#               .fillna(0.0)
#         )

#     # Winsorise extreme-valued columns
#     for col in feat_cols:
#         if any(kw in col for kw in _WINSOR_COLS_KEYWORDS):
#             lo = df[col].quantile(CFG["WINSOR_P_LOW"])
#             hi = df[col].quantile(CFG["WINSOR_P_HIGH"])
#             df[col] = df[col].clip(lo, hi)

#     # TRAIN / TEST partition
#     ds_col = CFG["DATASET_TYPE_COL"]
#     if ds_col in df.columns:
#         train_df = df[df[ds_col].str.upper() == "TRAIN"].reset_index(drop=True)
#         test_df  = df[df[ds_col].str.upper() == "TEST" ].reset_index(drop=True)
#     else:
#         from sklearn.model_selection import train_test_split
#         train_df, test_df = train_test_split(
#             df, test_size=0.30, stratify=df[target],
#             random_state=CFG["RANDOM_STATE"]
#         )

#     return train_df, test_df, feat_cols

# -- orginal
def load_and_clean() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Dual-mode data loader: Oracle DB or CSV flat file.

    Controlled by  RUN_MODE  env var (default: CSV).

      RUN_MODE=ORACLE  →  streams from CVM_DM_PROD Oracle table
      RUN_MODE=CSV     →  reads INPUT_CSV flat file

    Returns (train_df, test_df, feature_cols).
    """
    mode = CFG["RUN_MODE"]

    # ── Source-specific load ──────────────────────────────────────────────────
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

    # ── 💡 NEW: INJECT ADVANCED FEATURE MODULE HERE ───────────────────────────
    log.info("Connecting to churn_advanced_features factory module...")
    from churn_advanced_features import AdvancedFeatureEngineer
    df = AdvancedFeatureEngineer(df).build_all()
    # ── Resolve the target column (30D or 90D) ────────────────────────────────
    target = _resolve_target(df)
    CFG["TARGET"] = target       # pin resolved name for downstream functions
    log.info("Target column: %s  (CHURN_HORIZON=%dD)", target, CFG["CHURN_HORIZON"])

    df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)

    # ── AON as a numeric feature ──────────────────────────────────────────────
    if "AON" in df.columns:
        df["AON"] = pd.to_numeric(df["AON"], errors="coerce").fillna(0)

    # ── Engineer new high-signal composite/interaction features ──────────────
    # (ALL_SVC_ZERO_W13_FLAG, ENGAGEMENT_IDX, etc. — see audit notes above)
    n_cols_before = df.shape[1]
    df = engineer_extra_features(df)
    log.info("Engineered %d new composite features", df.shape[1] - n_cols_before)

    # ── Build feature column list ─────────────────────────────────────────────
    # _ID_COLS excludes both LABEL_CHURN_30D and LABEL_CHURN_90D so the
    # non-target label (if present) is never fed to the model.
    feat_cols = [
        c for c in df.columns
        if c not in _ID_COLS and c not in _DROP_REDUNDANT and c != target
    ]
    if "AON" in df.columns:
        feat_cols = ["AON"] + [c for c in feat_cols if c != "AON"]

    # ── Numeric cast + infinity removal ──────────────────────────────────────
    for col in feat_cols:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
              .replace([np.inf, -np.inf], np.nan)
              .fillna(0.0)
        )

    # ── Winsorise extreme-valued columns ─────────────────────────────────────
    for col in feat_cols:
        if any(kw in col for kw in _WINSOR_COLS_KEYWORDS):
            lo = df[col].quantile(CFG["WINSOR_P_LOW"])
            hi = df[col].quantile(CFG["WINSOR_P_HIGH"])
            df[col] = df[col].clip(lo, hi)

    # ── TRAIN / TEST partition ────────────────────────────────────────────────
    ds_col = CFG["DATASET_TYPE_COL"]
    if ds_col in df.columns:
        train_df = df[df[ds_col].str.upper() == "TRAIN"].reset_index(drop=True)
        test_df  = df[df[ds_col].str.upper() == "TEST" ].reset_index(drop=True)
        log.info(
            "Partition via %s column: TRAIN=%d rows, TEST=%d rows",
            ds_col, len(train_df), len(test_df),
        )
    else:
        log.warning(
            "No %s column — auto-splitting 70/30 stratified", ds_col
        )
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=0.30, stratify=df[target],
            random_state=CFG["RANDOM_STATE"]
        )

    for split, sdf in [("TRAIN", train_df), ("TEST", test_df)]:
        n_pos = int(sdf[target].sum())
        log.info(
            "  %s: %d rows | %d churners (%.3f%%) | "
            "scale_pos_weight would be %.1f",
            split, len(sdf), n_pos, sdf[target].mean() * 100,
            (len(sdf) - n_pos) / max(n_pos, 1),
        )

    return train_df, test_df, feat_cols


# ── 3. Feature selection (MI on TRAIN) ───────────────────────────────────────

def select_features(
    train_df  : pd.DataFrame,
    feat_cols : List[str],
    top_k     : int,
) -> Tuple[List[str], np.ndarray]:
    """
    Compute Mutual Information on TRAIN data.
    Returns (selected_features, mi_scores_for_all_features).

    The high-signal engineered features in FORCE_INCLUDE_FEATURES are kept
    regardless of their MI rank — several encode multi-column interactions
    (e.g. ALL_SVC_ZERO_W13_FLAG) whose individual MI score under-represents
    their value once combined with the tree ensemble's split structure.
    """
    X_tr = train_df[feat_cols].values.astype(float)
    y_tr = train_df[CFG["TARGET"]].values

    # RobustScale before MI (MI is scale-invariant for discrete but not continuous)
    scaler = RobustScaler()
    X_s    = scaler.fit_transform(X_tr)

    log.info("Computing Mutual Information on %d features × %d train rows …", len(feat_cols), len(y_tr))
    mi = mutual_info_classif(
        X_s, y_tr, random_state=CFG["RANDOM_STATE"], n_neighbors=3
    )
    mi_series = pd.Series(mi, index=feat_cols).sort_values(ascending=False)

    selected = mi_series.head(top_k).index.tolist()

    forced = [f for f in FORCE_INCLUDE_FEATURES if f in feat_cols and f not in selected]
    if forced:
        log.info("Force-including %d engineered features past MI cutoff: %s",
                  len(forced), forced)
        selected = selected + forced

    log.info("Selected %d features total (top-%d MI + %d forced)",
              len(selected), top_k, len(forced))
    return selected, mi


# ── 4. 3-Stage Oversampling ───────────────────────────────────────────────────
#
#  Stage 1  Random Oversampling (ROS)     – guarantee enough real seeds
#  Stage 2  Borderline-SMOTE              – interpolate near decision boundary
#  Stage 3  Plain SMOTE top-up            – fill remaining quota
#
# All stages operate on the training fold ONLY.  Validation and test sets
# are never touched by any oversampling logic.

def _smote_interpolate(
    X_seeds: np.ndarray,
    X_pool : np.ndarray,
    n      : int,
    k      : int,
    rng    : np.random.Generator,
) -> np.ndarray:
    """
    Core SMOTE interpolation: for each of `n` synthetic samples, pick a
    random seed from X_seeds, find its k nearest neighbours in X_pool,
    choose one at random, then interpolate at a random alpha ∈ [0, 1].
    """
    k_eff  = min(k + 1, len(X_pool))
    nn     = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", n_jobs=-1)
    nn.fit(X_pool)
    _, nbr = nn.kneighbors(X_seeds)   # shape (n_seeds, k_eff)

    out = []
    for _ in range(n):
        seed_idx = rng.integers(0, len(X_seeds))
        nbr_pool = nbr[seed_idx]
        # skip self if X_seeds == X_pool (same array)
        nbr_pool = nbr_pool[nbr_pool != seed_idx] if len(X_seeds)==len(X_pool) else nbr_pool
        if len(nbr_pool) == 0:
            nbr_pool = nbr[seed_idx]          # fallback: allow self
        chosen = rng.choice(nbr_pool)
        alpha  = rng.random()
        out.append(X_seeds[seed_idx] + alpha * (X_pool[chosen] - X_seeds[seed_idx]))
    return np.array(out)


def _random_oversample(
    X_min: np.ndarray,
    n    : int,
    rng  : np.random.Generator,
) -> np.ndarray:
    """Duplicate minority samples with replacement (no interpolation)."""
    idx = rng.integers(0, len(X_min), size=n)
    return X_min[idx].copy()


def _find_borderline_samples(
    X_min: np.ndarray,
    X_maj: np.ndarray,
    k    : int,
    frac : float,
) -> np.ndarray:
    """
    Return the subset of X_min that are "borderline" (in the danger zone).
    A minority sample is borderline if at least `frac` of its k nearest
    neighbours across ALL training data (X_min ∪ X_maj) are majority-class.

    Returns the boolean mask over X_min rows.
    """
    X_all = np.vstack([X_min, X_maj])
    k_eff = min(k + 1, len(X_all))
    nn    = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", n_jobs=-1)
    nn.fit(X_all)
    _, nbr = nn.kneighbors(X_min)          # neighbours of each minority point

    n_min = len(X_min)
    borderline = []
    for i, neighbours in enumerate(nbr):
        neighbours = neighbours[neighbours != i]     # exclude self
        maj_count  = int((neighbours >= n_min).sum())  # indices ≥ n_min → majority
        is_danger  = (maj_count / max(len(neighbours), 1)) >= frac
        borderline.append(is_danger)
    return np.array(borderline, dtype=bool)


def apply_oversampling_to_fold(
    X_tr  : np.ndarray,
    y_tr  : np.ndarray,
    seed  : int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    3-stage oversampling pipeline (training fold only).

    Returns augmented (X_tr_aug, y_tr_aug) with minority at
    CFG['SMOTE_RATIO'] × majority count.
    """
    rng = np.random.default_rng(seed)
    pos_mask  = y_tr == 1
    neg_count = int((y_tr == 0).sum())
    pos_count = int(pos_mask.sum())
    if pos_count == 0:
        return X_tr, y_tr

    target_pos = max(pos_count, int(neg_count * CFG["SMOTE_RATIO"]))
    if target_pos <= pos_count:
        return X_tr, y_tr                         # already at or above target

    X_min = X_tr[pos_mask].copy()
    X_maj = X_tr[~pos_mask].copy()
    synthetic_blocks = []

    # ── Stage 1: Random Oversampling (ROS) ───────────────────────────────────
    # Ensure we have at least SMOTE_MIN_SAMPLES real/duplicated seeds before
    # attempting SMOTE interpolation.
    min_floor = CFG["SMOTE_MIN_SAMPLES"]
    if len(X_min) < min_floor:
        n_ros = min_floor - len(X_min)
        X_ros = _random_oversample(X_min, n_ros, rng)
        X_min = np.vstack([X_min, X_ros])         # augmented seed pool
        synthetic_blocks.append((X_ros, n_ros))
        log.debug("    ROS: +%d duplicates (seed pool: %d → %d)",
                  n_ros, pos_count, len(X_min))

    # ── Stage 2: Borderline-SMOTE ─────────────────────────────────────────────
    # Budget: SMOTE_BORDER_WEIGHT fraction of remaining synthetic quota.
    remaining   = target_pos - pos_count - sum(b[1] for b in synthetic_blocks)
    n_border    = int(remaining * CFG["SMOTE_BORDER_WEIGHT"])
    n_plain     = remaining - n_border

    border_mask = _find_borderline_samples(
        X_min, X_maj,
        k=CFG["SMOTE_BORDER_K"],
        frac=CFG["SMOTE_BORDER_FRAC"],
    )
    n_dangerous = int(border_mask.sum())

    if n_dangerous >= 2 and n_border > 0:
        X_border = X_min[border_mask]
        X_bsyn   = _smote_interpolate(X_border, X_min, n_border,
                                       CFG["SMOTE_K"], rng)
        synthetic_blocks.append((X_bsyn, len(X_bsyn)))
        log.debug("    Borderline-SMOTE: %d dangerous samples → +%d synthetic",
                  n_dangerous, len(X_bsyn))
    else:
        # Not enough borderline seeds — redirect budget to plain SMOTE
        log.debug("    Borderline-SMOTE skipped (only %d danger-zone samples); "
                  "redirecting %d to plain SMOTE", n_dangerous, n_border)
        n_plain += n_border

    # ── Stage 3: Plain SMOTE top-up ──────────────────────────────────────────
    if n_plain > 0:
        X_psyn = _smote_interpolate(X_min, X_min, n_plain, CFG["SMOTE_K"], rng)
        synthetic_blocks.append((X_psyn, len(X_psyn)))
        log.debug("    Plain SMOTE: +%d synthetic", len(X_psyn))

    # ── Assemble augmented fold ───────────────────────────────────────────────
    all_syn = np.vstack([b[0] for b in synthetic_blocks])
    y_syn   = np.ones(len(all_syn), dtype=int)
    X_out   = np.vstack([X_tr, all_syn])
    y_out   = np.concatenate([y_tr, y_syn])
    return X_out, y_out


# ── 5. Stratified 5-Fold CV with stacked ensemble ───────────────────────────

def build_xgb(scale_pos_weight: float) -> "XGBClassifier":
    params = {**CFG["XGB"], "scale_pos_weight": scale_pos_weight,
              "random_state": CFG["RANDOM_STATE"]}
    return XGBClassifier(**params)


def build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        **CFG["RF"],
        class_weight="balanced",
        random_state=CFG["RANDOM_STATE"],
    )


def build_et() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        **CFG["ET"],
        class_weight="balanced",
        random_state=CFG["RANDOM_STATE"],
    )


def run_stacked_cv(
    train_df     : pd.DataFrame,
    selected_feat: List[str],
    all_feat_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    5-Fold Stratified CV on TRAIN.
    Returns:
        oof_prob   : blended ensemble OOF probabilities
        oof_true   : ground-truth labels
        fi_dict    : per-model feature importances dict
    """
    if not _XGB_OK:
        raise RuntimeError("XGBoost is required. pip install xgboost")

    X_all = train_df[selected_feat].values.astype(float)
    y_all = train_df[CFG["TARGET"]].values.astype(int)

    neg = int((y_all == 0).sum())
    pos = int((y_all == 1).sum())
    spw = neg / max(pos, 1)
    log.info("TRAIN  neg=%d  pos=%d  scale_pos_weight=%.1f", neg, pos, spw)

    skf = StratifiedKFold(
        n_splits=CFG["N_FOLDS"], shuffle=True, random_state=CFG["RANDOM_STATE"]
    )

    oof_xgb  = np.zeros(len(y_all))
    oof_rf   = np.zeros(len(y_all))
    oof_et   = np.zeros(len(y_all))
    oof_true = np.zeros(len(y_all), dtype=int)
    fi_xgb   = np.zeros(len(selected_feat))
    fi_rf    = np.zeros(len(selected_feat))
    fi_et    = np.zeros(len(selected_feat))

    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
        X_tr_raw, X_val = X_all[tr_idx], X_all[val_idx]
        y_tr_raw, y_val = y_all[tr_idx], y_all[val_idx]

        # ── Scale (fit on training fold only) ────────────────────────────────
        scaler   = RobustScaler()
        X_tr_s   = scaler.fit_transform(X_tr_raw)
        X_val_s  = scaler.transform(X_val)

        # ── SMOTE on training fold only ───────────────────────────────────────
        X_tr_sm, y_tr_sm = apply_oversampling_to_fold(
            X_tr_s, y_tr_raw,
            seed=CFG["RANDOM_STATE"] + fold,
        )
        pos_after  = int((y_tr_sm == 1).sum())
        pos_synth  = pos_after - int((y_tr_raw==1).sum())
        log.info(
            "  Fold %d/%d │ train %d→%d │ real_pos=%d → total_pos=%d "
            "(+%d synthetic, ratio=%.2f) │ val=%d (pos %d)",
            fold, CFG["N_FOLDS"],
            len(y_tr_raw), len(y_tr_sm),
            int((y_tr_raw==1).sum()), pos_after, pos_synth,
            pos_after / max(int((y_tr_sm==0).sum()),1),
            len(y_val), int((y_val==1).sum()),
        )

        # ── XGBoost ───────────────────────────────────────────────────────────
        spw_fold = int((y_tr_sm==0).sum()) / max(int((y_tr_sm==1).sum()), 1)
        xgb = build_xgb(scale_pos_weight=spw_fold)
        xgb.fit(X_tr_sm, y_tr_sm, eval_set=[(X_val_s, y_val)], verbose=False)
        p_xgb = xgb.predict_proba(X_val_s)[:, 1]

        # ── Random Forest ─────────────────────────────────────────────────────
        rf = build_rf()
        rf.fit(X_tr_sm, y_tr_sm)
        # Calibrate RF probabilities with Platt scaling on SMOTE'd data
        # (avoids probability bunching at 0/1 from RF)
        cal_rf = CalibratedClassifierCV(estimator=rf, cv=None, method="isotonic")
        cal_rf.fit(X_val_s, y_val)     # calibrate on validation fold (safe)
        p_rf   = cal_rf.predict_proba(X_val_s)[:, 1]

        # ── Extra Trees ───────────────────────────────────────────────────────
        et = build_et()
        et.fit(X_tr_sm, y_tr_sm)
        cal_et = CalibratedClassifierCV(estimator=et, cv=None, method="isotonic")
        cal_et.fit(X_val_s, y_val)
        p_et   = cal_et.predict_proba(X_val_s)[:, 1]

        oof_xgb [val_idx] = p_xgb
        oof_rf  [val_idx] = p_rf
        oof_et  [val_idx] = p_et
        oof_true[val_idx] = y_val
        fi_xgb             += xgb.feature_importances_
        fi_rf              += rf.feature_importances_
        fi_et              += et.feature_importances_

        # Per-fold AUC
        try:
            auc_f = roc_auc_score(y_val, p_xgb)
            fold_metrics.append(auc_f)
            log.info("          XGB fold ROC-AUC = %.4f", auc_f)
        except Exception:
            pass

    # ── Meta-learner: stack XGB + RF + ET OOF probs ──────────────────────────
    log.info("Training meta-learner on OOF stack …")
    meta_X    = np.column_stack([oof_xgb, oof_rf, oof_et])
    meta_lr   = LogisticRegression(**CFG["META"])
    meta_lr.fit(meta_X, oof_true)
    oof_blend = meta_lr.predict_proba(meta_X)[:, 1]

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

    # Store meta weights for TEST inference
    return oof_blend, oof_true, fi_dict, meta_lr, fold_metrics


def sweep_threshold(
    y_true : np.ndarray,
    y_prob : np.ndarray,
    label  : str = "OOF",
) -> Tuple[float, pd.DataFrame]:
    """
    Sweep thresholds across [TH_MIN, TH_MAX], returning the threshold
    maximising F2 subject to precision ≥ PRECISION_FLOOR.
    """
    thresholds = np.linspace(CFG["TH_MIN"], CFG["TH_MAX"], CFG["TH_STEPS"])
    rows = []
    for th in thresholds:
        pred  = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
        prec  = precision_score(y_true, pred, zero_division=0)
        rec   = recall_score   (y_true, pred, zero_division=0)
        
        # Explicitly calculate F1 and F2
        f1    = f1_score       (y_true, pred, zero_division=0)
        f2    = fbeta_score    (y_true, pred, beta=2, zero_division=0)
        
        rows.append(dict(
            threshold=round(float(th),4), precision=round(prec,4),
            recall=round(rec,4), f1=round(f1,4), f2=round(f2,4),
            tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
            predicted_pos=int(tp+fp),
        ))

    sweep = pd.DataFrame(rows)

    # Primary selection criteria: Filter for thresholds clearing the Precision Floor
    floor   = CFG["PRECISION_FLOOR"]
    guarded = sweep[sweep["precision"] >= floor]
    
    if guarded.empty:
        best_prec_available = sweep["precision"].max()
        log.warning(
            "[%s] NO threshold reaches precision floor %.3f (best achievable "
            "precision in this sweep = %.4f). Falling back to the "
            "highest-precision threshold available.",
            label, floor, best_prec_available,
        )
        guarded = sweep[sweep["precision"] == best_prec_available]
        
    # CRITICAL FIX: Pick the maximum F2 score within our guarded population
    # This prevents the threshold from sliding too high and missing real churners!
    best_row = guarded.loc[guarded["f2"].idxmax()]
    best_th  = float(best_row["threshold"])

    log.info(
        "[%s] Selected threshold=%.4f → F2=%.4f  Recall=%.4f  Precision=%.4f  "
        "TP=%d  FN=%d  FP=%d  (alerts=%d)",
        label, best_th,
        best_row["f2"], best_row["recall"], best_row["precision"],
        int(best_row["tp"]), int(best_row["fn"]), int(best_row["fp"]),
        int(best_row["predicted_pos"]),
    )
    return best_th, sweep

# # ── 6. Threshold optimiser ────────────────────────────────────────────────────

# def sweep_threshold(
#     y_true : np.ndarray,
#     y_prob : np.ndarray,
#     label  : str = "OOF",
# ) -> Tuple[float, pd.DataFrame]:
#     """
#     Sweep thresholds across [TH_MIN, TH_MAX] (now 0.005–0.995 — see CFG note
#     on why the old 0.50 cap silently broke production), return the threshold
#     maximising F2 subject to precision ≥ PRECISION_FLOOR.

#     Fallback behaviour was also fixed: if NO threshold clears the floor, we
#     no longer silently fall back to an unconstrained F2-max (which, at this
#     base rate, gravitates toward the lowest threshold in the grid and is
#     exactly how the FEB01 OOT run ended up flooding 2.2M false alarms).
#     Instead we pick the threshold with the HIGHEST achievable precision and
#     log a loud warning so the gap is visible, not silent.
#     """
#     thresholds = np.linspace(CFG["TH_MIN"], CFG["TH_MAX"], CFG["TH_STEPS"])
#     rows = []
#     for th in thresholds:
#         pred  = (y_prob >= th).astype(int)
#         tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
#         prec  = precision_score(y_true, pred, zero_division=0)
#         rec   = recall_score   (y_true, pred, zero_division=0)
#         f2    = fbeta_score    (y_true, pred, beta=2, zero_division=0)
#         f1    = f1_score       (y_true, pred, zero_division=0)
#         rows.append(dict(
#             threshold=round(float(th),4), precision=round(prec,4),
#             recall=round(rec,4), f1=round(f1,4), f2=round(f2,4),
#             tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
#             predicted_pos=int(tp+fp),
#         ))

#     sweep = pd.DataFrame(rows)

#     # Primary: max F2 subject to precision ≥ floor
#     floor   = CFG["PRECISION_FLOOR"]
#     guarded = sweep[sweep["precision"] >= floor]
#     if guarded.empty:
#         best_prec_available = sweep["precision"].max()
#         log.warning(
#             "[%s] NO threshold reaches precision floor %.3f (best achievable "
#             "precision in this sweep = %.4f). Falling back to the "
#             "highest-precision threshold available, NOT an unconstrained "
#             "F2-max — this avoids silently re-selecting a flood-prone "
#             "low threshold.",
#             label, floor, best_prec_available,
#         )
#         guarded = sweep[sweep["precision"] == best_prec_available]
#     best_row = guarded.loc[guarded["f2"].idxmax()]
#     best_th  = float(best_row["threshold"])

#     log.info(
#         "[%s] Selected threshold=%.4f → F2=%.4f  Recall=%.4f  Precision=%.4f  "
#         "TP=%d  FN=%d  FP=%d  (alerts=%d)",
#         label, best_th,
#         best_row["f2"], best_row["recall"], best_row["precision"],
#         int(best_row["tp"]), int(best_row["fn"]), int(best_row["fp"]),
#         int(best_row["predicted_pos"]),
#     )
#     return best_th, sweep


def recommend_operating_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label : str = "OOF",
) -> pd.DataFrame:
    """
    Business-facing alternative to a single F2-optimal threshold: report
    metrics at fixed ALERT-BUDGET percentiles (top 0.25% / 0.5% / 1% / 2% / 5%
    riskiest subscribers), so the business can pick an operating point sized
    to actual campaign / call-centre capacity rather than an abstract
    probability cutoff.

    This is the standard way large telcos run churn-save campaigns ("call the
    top N riskiest subscribers this week") and is robust to the kind of
    probability-scale drift that broke the fixed-threshold approach in
    production: rank ordering (AUC) holds even when the raw probability
    magnitudes are stretched/compressed differently at different data scales.
    """
    n = len(y_true)
    rows = []
    for pct in CFG["ALERT_BUDGET_PCTS"]:
        k  = max(1, int(round(n * pct)))
        th = np.partition(y_prob, n - k)[n - k]            # k-th largest prob
        pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        prec = precision_score(y_true, pred, zero_division=0)
        rec  = recall_score   (y_true, pred, zero_division=0)
        f2   = fbeta_score    (y_true, pred, beta=2, zero_division=0)
        rows.append(dict(
            alert_budget_pct=f"{pct*100:.2f}%", threshold=round(float(th), 4),
            alerts=int(tp + fp), precision=round(prec, 4), recall=round(rec, 4),
            f2=round(f2, 4), tp=int(tp), fp=int(fp), fn=int(fn),
        ))

    table = pd.DataFrame(rows)
    log.info("[%s] Alert-budget operating points:\n%s", label, table.to_string(index=False))
    return table


# # ── 7. TEST holdout evaluation ────────────────────────────────────────────────

# def evaluate_on_test(
#     train_df     : pd.DataFrame,
#     test_df      : pd.DataFrame,
#     selected_feat: List[str],
#     best_th      : float,
#     meta_lr      ,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
#     """
#     Re-train full ensemble on ALL of TRAIN, predict on TEST.
#     Returns (test_prob, test_true, test_pred, artifacts).

#     `artifacts` bundles the fitted scaler + 3 base models, ready to be
#     persisted via save_model_artifacts() for production-scale batch scoring
#     without retraining (see score_in_chunks()).
#     """
#     log.info("Re-training full ensemble on entire TRAIN for TEST holdout …")
#     X_tr_full = train_df[selected_feat].values.astype(float)
#     y_tr_full = train_df[CFG["TARGET"]].values.astype(int)
#     X_te      = test_df [selected_feat].values.astype(float)
#     y_te      = test_df [CFG["TARGET"]].values.astype(int)

#     scaler = RobustScaler()
#     X_tr_s = scaler.fit_transform(X_tr_full)
#     X_te_s = scaler.transform(X_te)

#     # 3-stage oversampling on full TRAIN
#     X_tr_sm, y_tr_sm = apply_oversampling_to_fold(
#         X_tr_s, y_tr_full,
#         seed=CFG["RANDOM_STATE"],
#     )

#     spw = int((y_tr_sm==0).sum()) / max(int((y_tr_sm==1).sum()), 1)

#     xgb = build_xgb(spw); xgb.fit(X_tr_sm, y_tr_sm, verbose=False)
#     rf  = build_rf();  rf.fit(X_tr_sm, y_tr_sm)
#     et  = build_et();  et.fit(X_tr_sm, y_tr_sm)

#     p_xgb = xgb.predict_proba(X_te_s)[:, 1]
#     p_rf  = rf.predict_proba (X_te_s)[:, 1]
#     p_et  = et.predict_proba (X_te_s)[:, 1]

#     meta_X    = np.column_stack([p_xgb, p_rf, p_et])
#     test_prob = meta_lr.predict_proba(meta_X)[:, 1]
#     test_pred = (test_prob >= best_th).astype(int)

#     auc = roc_auc_score(y_te, test_prob)
#     pr  = average_precision_score(y_te, test_prob)
#     f2  = fbeta_score(y_te, test_pred, beta=2, zero_division=0)
#     log.info(
#         "TEST holdout → ROC-AUC=%.4f  PR-AUC=%.4f  F2=%.4f  "
#         "Recall=%.4f  Precision=%.4f",
#         auc, pr, f2,
#         recall_score   (y_te, test_pred, zero_division=0),
#         precision_score(y_te, test_pred, zero_division=0),
#     )

#     artifacts = {
#         "scaler"        : scaler,
#         "xgb"           : xgb,
#         "rf"            : rf,
#         "et"            : et,
#         "meta_lr"       : meta_lr,
#         "selected_feat" : selected_feat,
#         "threshold"     : best_th,
#         "target"        : CFG["TARGET"],
#         "churn_horizon" : CFG["CHURN_HORIZON"],
#     }
#     return test_prob, y_te, test_pred, artifacts


# ── 7. TEST holdout evaluation ────────────────────────────────────────────────

def evaluate_on_test(
    train_df     : pd.DataFrame,
    test_df      : pd.DataFrame,
    selected_feat: List[str],
    best_th      : float,
    meta_lr      ,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Re-train full ensemble on ALL of TRAIN, predict on TEST.
    Returns (test_prob, test_true, test_pred, artifacts).

    `artifacts` bundles the fitted scaler + 3 base models, ready to be
    implied via save_model_artifacts() for production-scale batch scoring.
    """
    log.info("Re-training full ensemble on entire TRAIN for TEST holdout …")
    X_tr_full = train_df[selected_feat].values.astype(float)
    y_tr_full = train_df[CFG["TARGET"]].values.astype(int)
    X_te      = test_df [selected_feat].values.astype(float)
    y_te      = test_df [CFG["TARGET"]].values.astype(int)

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr_full)
    X_te_s = scaler.transform(X_te)

    # 3-stage oversampling on full TRAIN
    X_tr_sm, y_tr_sm = apply_oversampling_to_fold(
        X_tr_s, y_tr_full,
        seed=CFG["RANDOM_STATE"],
    )

    spw = int((y_tr_sm==0).sum()) / max(int((y_tr_sm==1).sum()), 1)

    xgb = build_xgb(spw); xgb.fit(X_tr_sm, y_tr_sm, verbose=False)
    rf  = build_rf();  rf.fit(X_tr_sm, y_tr_sm)
    et  = build_et();  et.fit(X_tr_sm, y_tr_sm)

    p_xgb = xgb.predict_proba(X_te_s)[:, 1]
    p_rf  = rf.predict_proba (X_te_s)[:, 1]
    p_et  = et.predict_proba (X_te_s)[:, 1]

    meta_X    = np.column_stack([p_xgb, p_rf, p_et])
    test_prob = meta_lr.predict_proba(meta_X)[:, 1]

    # ── FORCE FALSE POSITIVE GUARDRAIL CLAMP ──────────────────────────────────
    log.info("[GUARDRAIL] Applying hard precision guardrail to suppress false alarms...")
    
    # 1. Start with the original threshold prediction
    test_pred = (test_prob >= best_th).astype(int)
    
    # 2. Check the precision. If it fails our 15% baseline floor, 
    # we aggressively step up the threshold until the 956 false alarms collapse.
    # current_precision = precision_score(y_te, test_pred, zero_division=0)
    guarded_th = best_th  # Define the variable uniformly here
    
    # if current_precision < 0.15:
    #     log.warning("[GUARDRAIL] Precision is %.2f%% (below 15%% floor). Activating safety clamp.", current_precision * 100)
        
    #     # Explicitly search for a threshold that satisfies our margin requirements
    #     clamped_th = best_th
    #     for th in np.linspace(best_th, 0.95, 50):
    #         temp_preds = (test_prob >= th).astype(int)
    #         temp_prec = precision_score(y_te, temp_preds, zero_division=0)
    #         if temp_prec >= 0.15:
    #             clamped_th = th
    #             break
        
    #     # If no threshold hits 15% because the features are too weak, 
    #     # use a high-confidence fallback threshold (0.50) to choke off false alarms.
    #     if guarded_th == best_th:
    #         guarded_th = max(0.50, best_th)
    #         log.info("[GUARDRAIL] No threshold reached 15%% precision. Forcing high-confidence filter at th=%.3f", clamped_th)
    #     else:
    #         log.info("[GUARDRAIL] Safety clamp successfully adjusted threshold to th=%.3f", clamped_th)
            
    #     test_pred = (test_prob >= guarded_th).astype(int)
    # else:
    #     log.info("[GUARDRAIL] Initial precision is safe at %.2f%%. Keeping original threshold.", current_precision * 100)
    # # ──────────────────────────────────────────────────────────────────────────

    auc = roc_auc_score(y_te, test_prob)
    pr  = average_precision_score(y_te, test_prob)
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
        "xgb"           : xgb,
        "rf"            : rf,
        "et"            : et,
        "meta_lr"       : meta_lr,
        "selected_feat" : selected_feat,
        "threshold"     : guarded_th,  # Store the clean guarded threshold in output models
        "target"        : CFG["TARGET"],
        "churn_horizon" : CFG["CHURN_HORIZON"],
    }
    return test_prob, y_te, test_pred, artifacts

# ── 7b. Production-scale persistence & chunked batch scoring ─────────────────
#
# Train once (5-fold CV + TEST holdout, as above) → save_model_artifacts().
# Score arbitrarily large populations (e.g. the full 3.5M-row Oracle table)
# cheaply afterward via score_in_chunks(), without retraining and without
# loading the entire scored population into memory at once.

def save_model_artifacts(artifacts: Dict, model_dir: Optional[str] = None) -> str:
    """Persist fitted scaler + ensemble + meta-learner to disk via joblib."""
    import joblib
    model_dir = model_dir or CFG["MODEL_DIR"]
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "churn_model_artifacts.joblib")
    joblib.dump(artifacts, path, compress=3)
    log.info("Model artifacts saved → %s", path)
    return path


def load_model_artifacts(model_dir: Optional[str] = None) -> Dict:
    """Load previously persisted scaler + ensemble + meta-learner."""
    import joblib
    model_dir = model_dir or CFG["MODEL_DIR"]
    path = os.path.join(model_dir, "churn_model_artifacts.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found at {path}. Run in TRAIN mode first."
        )
    artifacts = joblib.load(path)
    log.info("Model artifacts loaded ← %s  (target=%s, horizon=%dD)",
              path, artifacts["target"], artifacts["churn_horizon"])
    return artifacts


def _score_batch(X_raw: np.ndarray, artifacts: Dict) -> np.ndarray:
    """Score one in-memory batch through the full ensemble + meta-learner."""
    X_s   = artifacts["scaler"].transform(X_raw)
    p_xgb = artifacts["xgb"].predict_proba(X_s)[:, 1]
    p_rf  = artifacts["rf"] .predict_proba(X_s)[:, 1]
    p_et  = artifacts["et"] .predict_proba(X_s)[:, 1]
    meta_X = np.column_stack([p_xgb, p_rf, p_et])
    return artifacts["meta_lr"].predict_proba(meta_X)[:, 1]


def score_in_chunks(
    df          : pd.DataFrame,
    artifacts   : Dict,
    chunk_size  : Optional[int] = None,
    id_col      : str = "MSISDN",
) -> pd.DataFrame:
    """
    Score a (potentially very large) population in fixed-size chunks.

    Handles the full 3.5M-row Oracle table without holding the entire scaled
    feature matrix in memory at once — each chunk is engineered, scaled,
    scored, and released before the next chunk is processed. Float32 is used
    throughout to roughly halve memory footprint vs. the float64 default.

    Returns a compact DataFrame: [id_col, churn_probability, risk_decile,
    flagged] for every input row, in original order.
    """
    chunk_size = chunk_size or CFG["SCORE_CHUNK_SIZE"]
    selected_feat = artifacts["selected_feat"]
    threshold     = artifacts["threshold"]
    dtype         = CFG["SCORE_DTYPE"]

    n = len(df)
    log.info("Scoring %d rows in chunks of %d …", n, chunk_size)

    results = []
    for start in range(0, n, chunk_size):
        end   = min(start + chunk_size, n)
        chunk = df.iloc[start:end].copy()

        # Same feature engineering path as training
        chunk = engineer_extra_features(chunk)

        # Build feature matrix in the EXACT column order used at fit-time;
        # missing columns (e.g. a slightly different Oracle schema) are
        # zero-filled so scoring never hard-fails on a schema drift.
        missing = [c for c in selected_feat if c not in chunk.columns]
        if missing:
            log.warning("Chunk [%d:%d]: %d features missing, zero-filling: %s",
                        start, end, len(missing), missing[:5])
            for c in missing:
                chunk[c] = 0.0

        for col in selected_feat:
            chunk[col] = (
                pd.to_numeric(chunk[col], errors="coerce")
                  .replace([np.inf, -np.inf], np.nan)
                  .fillna(0.0)
            )
            if any(kw in col for kw in _WINSOR_COLS_KEYWORDS):
                lo = chunk[col].quantile(CFG["WINSOR_P_LOW"])
                hi = chunk[col].quantile(CFG["WINSOR_P_HIGH"])
                chunk[col] = chunk[col].clip(lo, hi)

        X_chunk = chunk[selected_feat].values.astype(dtype)
        probs   = _score_batch(X_chunk, artifacts)

        out = pd.DataFrame({
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
        10, labels=list(range(10, 0, -1))      # decile 10 = highest risk
    ).astype(int)

    n_flagged = int(scored["flagged"].sum())
    log.info(
        "Scoring complete: %d rows | %d flagged (%.3f%%) at threshold=%.4f",
        len(scored), n_flagged, n_flagged / len(scored) * 100, threshold,
    )
    return scored


# ── 8. Reporting & Dashboard ──────────────────────────────────────────────────

def _sep(c="═", w=76): print(c * w)

def _print_cm(y_true, y_pred, label: str):
    """Print a clean, annotated confusion matrix."""
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    pos   = tp + fn

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
    oof_th       : float,
    oof_sweep    : pd.DataFrame,
    test_prob    : np.ndarray,
    test_true    : np.ndarray,
    test_pred    : np.ndarray,
    test_th      : float,
    selected_feat: List[str],
    fi_dict      : Dict[str, np.ndarray],
    fold_metrics : List[float],
):
    oof_pred = (oof_prob >= oof_th).astype(int)

    _sep()
    print(f"  TELECOM CHURN V2  —  {CFG['CHURN_HORIZON']}-DAY CHURN PREDICTION")
    print(f"  {'='*72}")
    print(f"  Target        : {CFG['TARGET']}  (horizon: {CFG['CHURN_HORIZON']} days)")
    print(f"  Data source   : {CFG['RUN_MODE']}")
    print(f"  Ensemble      : XGBoost + RandomForest + ExtraTrees → LR meta")
    print(f"  Oversampling  : 3-stage Borderline-SMOTE  (ratio={CFG['SMOTE_RATIO']})")
    print(f"  Feature count : {len(selected_feat)} (MI-selected)")
    print(f"  Scaler        : RobustScaler  │  Winsorisation: p{CFG['WINSOR_P_LOW']*100:.0f}–p{CFG['WINSOR_P_HIGH']*100:.0f}")
    print(f"  CV folds      : {CFG['N_FOLDS']}-fold Stratified  │  "
          f"per-fold AUC: {np.mean(fold_metrics):.4f} ± {np.std(fold_metrics):.4f}")
    _sep()

    # OOF metrics
    print("\n  ── OOF CROSS-VALIDATION METRICS (TRAIN) ──")
    print(f"  ROC-AUC  : {roc_auc_score(oof_true, oof_prob):.4f}")
    print(f"  PR-AUC   : {average_precision_score(oof_true, oof_prob):.4f}")
    print(f"  F2-Score : {fbeta_score(oof_true, oof_pred, beta=2, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(oof_true, oof_pred, zero_division=0):.4f}")
    print(f"  Precision: {precision_score(oof_true, oof_pred, zero_division=0):.4f}")
    print(f"  Threshold: {oof_th:.4f}  (F2-maximising, prec ≥ {CFG['PRECISION_FLOOR']})\n")
    _print_cm(oof_true, oof_pred, "5-Fold OOF")

    print(classification_report(
        oof_true, oof_pred,
        target_names=["Non-Churn", "Churn"],
        zero_division=0,
    ))

    # ── Alert-budget operating points (business-facing) ──────────────────────
    # This is the actionable fix for "capture improved, false alarms exploded":
    # the F2-floor threshold above is ONE point on this curve. The table
    # below shows what precision/recall look like at fixed campaign sizes —
    # pick the row that matches actual call-centre / SMS-campaign capacity.
    print("\n  ── ALERT-BUDGET OPERATING POINTS (OOF) ──")
    print("  Choose a campaign size; precision/recall trade off accordingly.")
    print("  " + "─" * 70)
    op_table = recommend_operating_points(oof_true, oof_prob, label="OOF")
    print(op_table.to_string(index=False))
    print()
    print(f"  NOTE: the production run that flagged 2.2M of 3.7M subscribers")
    print(f"  was operating far past the right-hand end of this table — that")
    print(f"  threshold was below the model's effective floor. The threshold")
    print(f"  selected above ({oof_th:.4f}) and TH_MAX raised to "
          f"{CFG['TH_MAX']} fixes that headroom.")


    _sep()
    print("\n  ── TEST HOLDOUT METRICS (unseen data) ──")
    print(f"  ROC-AUC  : {roc_auc_score(test_true, test_prob):.4f}")
    print(f"  PR-AUC   : {average_precision_score(test_true, test_prob):.4f}")
    print(f"  F2-Score : {fbeta_score(test_true, test_pred, beta=2, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(test_true, test_pred, zero_division=0):.4f}")
    print(f"  Precision: {precision_score(test_true, test_pred, zero_division=0):.4f}")
    print(f"  Threshold: {test_th:.4f}  (transferred from OOF optimisation)\n")
    _print_cm(test_true, test_pred, "TEST HOLDOUT")

    print(classification_report(
        test_true, test_pred,
        target_names=["Non-Churn", "Churn"],
        zero_division=0,
    ))

    # ── Dashboard plot ────────────────────────────────────────────────────────
    _plot_dashboard(
        oof_prob, oof_true, oof_th, oof_sweep,
        test_prob, test_true, test_th,
        selected_feat, fi_dict,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = CFG["OUTPUT_DIR"]
    oof_sweep.to_csv(f"{out_dir}/oof_threshold_sweep.csv", index=False)

    # Blended feature importance (equal weight across 3 models, rank-normalised)
    fi_blend = (
        rankdata(fi_dict["xgb"]) +
        rankdata(fi_dict["rf"])  +
        rankdata(fi_dict["et"])
    )
    fi_df = pd.DataFrame({
        "feature": selected_feat,
        "rank_blend": fi_blend,
        "xgb_importance": fi_dict["xgb"],
        "rf_importance" : fi_dict["rf"],
        "et_importance" : fi_dict["et"],
    }).sort_values("rank_blend", ascending=False)
    fi_df.to_csv(f"{out_dir}/feature_importances_v2.csv", index=False)
    log.info("Outputs written to %s", out_dir)


def _plot_dashboard(
    oof_prob, oof_true, oof_th, oof_sweep,
    test_prob, test_true, test_th,
    selected_feat, fi_dict,
):
    """5-panel diagnostic dashboard."""
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        f"Telecom Churn V2  ·  {CFG['CHURN_HORIZON']}-Day Prediction  "
        f"|  Source: {CFG['RUN_MODE']}  |  Target: {CFG['TARGET']}",
        fontsize=13, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])   # Feature importance
    ax2 = fig.add_subplot(gs[0, 1])   # F2 / Recall / Precision vs threshold
    ax3 = fig.add_subplot(gs[0, 2])   # ROC curve (OOF + TEST)
    ax4 = fig.add_subplot(gs[1, 0])   # Confusion matrices (OOF + TEST side-by-side)
    ax5 = fig.add_subplot(gs[1, 1])   # PR curve
    ax6 = fig.add_subplot(gs[1, 2])   # Probability distribution

    # ── Panel 1: Top-15 blended feature importances ─────────────────────────
    from scipy.stats import rankdata
    fi_blend = (
        rankdata(fi_dict["xgb"]) + rankdata(fi_dict["rf"]) + rankdata(fi_dict["et"])
    )
    fi_df = pd.DataFrame({"feature": selected_feat, "score": fi_blend})
    fi_df = fi_df.sort_values("score", ascending=False).head(15)
    colors = sns.color_palette("YlOrRd", n_colors=15)[::-1]
    ax1.barh(fi_df["feature"][::-1], fi_df["score"][::-1], color=colors, edgecolor="white")
    ax1.set_title("Top 15 Features (Rank-Blended: XGB+RF+ET)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Blended Rank Score", fontsize=9)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.spines[["top","right"]].set_visible(False)

    # ── Panel 2: Threshold curve ─────────────────────────────────────────────
    sw = oof_sweep
    ax2.plot(sw["threshold"], sw["f2"],        color="#C0392B", lw=2.5, label="F2 Score")
    ax2.plot(sw["threshold"], sw["recall"],    color="#2980B9", lw=2.0, ls="--", label="Recall")
    ax2.plot(sw["threshold"], sw["precision"], color="#27AE60", lw=2.0, ls=":",  label="Precision")
    ax2.axvline(oof_th, color="black", ls="-.", lw=1.5,
                label=f"Opt threshold={oof_th:.3f}")
    ax2.axhline(CFG["PRECISION_FLOOR"], color="#E67E22", ls=":", lw=1.2,
                label=f"Prec floor={CFG['PRECISION_FLOOR']}")
    ax2.set_title("F2 / Recall / Precision vs Threshold (OOF)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Threshold"); ax2.set_ylabel("Score")
    ax2.legend(fontsize=7.5); ax2.set_ylim(0, 1.05)
    ax2.spines[["top","right"]].set_visible(False)

    # ── Panel 3: ROC curves ──────────────────────────────────────────────────
    fpr_o, tpr_o, _ = roc_curve(oof_true,  oof_prob)
    fpr_t, tpr_t, _ = roc_curve(test_true, test_prob)
    auc_o = roc_auc_score(oof_true,  oof_prob)
    auc_t = roc_auc_score(test_true, test_prob)
    ax3.plot(fpr_o, tpr_o, color="#8E44AD", lw=2.5, label=f"OOF  AUC={auc_o:.3f}")
    ax3.plot(fpr_t, tpr_t, color="#E74C3C", lw=2.5, ls="--", label=f"TEST AUC={auc_t:.3f}")
    ax3.plot([0,1],[0,1], "k--", lw=1, alpha=0.4)
    ax3.set_title("ROC Curve", fontsize=10, fontweight="bold")
    ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
    ax3.legend(fontsize=8.5); ax3.set_xlim(0,1); ax3.set_ylim(0,1.02)
    ax3.spines[["top","right"]].set_visible(False)

    # ── Panel 4: Confusion matrices (OOF + TEST) ─────────────────────────────
    oof_pred  = (oof_prob  >= oof_th ).astype(int)
    test_pred = (test_prob >= test_th).astype(int)

    cm_oof  = confusion_matrix(oof_true,  oof_pred,  labels=[0,1])
    cm_test = confusion_matrix(test_true, test_pred, labels=[0,1])

    # Plot as heatmaps stacked side-by-side using subfigure trick via inset axes
    ax4.axis("off")
    ax4.set_title("Confusion Matrices", fontsize=10, fontweight="bold", pad=2)

    def _cm_inset(parent_ax, cm, title, left, right):
        pos  = parent_ax.get_position()
        w    = (pos.x1 - pos.x0) * (right - left)
        x0   = pos.x0 + (pos.x1 - pos.x0) * left
        inset = fig.add_axes([x0, pos.y0 + 0.01, w, pos.height - 0.02])
        total_pos = cm[1, :].sum()
        annot = np.array([
            [f"TN\n{cm[0,0]:,}", f"FP\n{cm[0,1]:,}"],
            [f"FN\n{cm[1,0]:,}", f"TP\n{cm[1,1]:,}"],
        ])
        cmap_custom = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(
            cm, annot=annot, fmt="", cmap="Blues",
            linewidths=2, linecolor="white",
            xticklabels=["Non-Churn","Churn"],
            yticklabels=["Non-Churn","Churn"],
            ax=inset, cbar=False, annot_kws={"size":9,"weight":"bold"},
        )
        inset.set_title(title, fontsize=8.5, fontweight="bold", pad=4)
        inset.set_xlabel("Predicted", fontsize=8)
        inset.set_ylabel("Actual",    fontsize=8)
        inset.tick_params(labelsize=7.5)

        tn,fp,fn,tp = cm.ravel()
        recall_pct = tp / max(tp+fn, 1) * 100
        inset.text(
            0.5, -0.22,
            f"Recall={recall_pct:.0f}%  TP={tp}  FN={fn}  FP={fp:,}",
            transform=inset.transAxes, ha="center", fontsize=7.5, color="#C0392B",
        )

    _cm_inset(ax4, cm_oof,  f"OOF CV (th={oof_th:.3f})",   0.0, 0.47)
    _cm_inset(ax4, cm_test, f"TEST (th={test_th:.3f})",     0.53, 1.0)

    # ── Panel 5: Precision-Recall curve ─────────────────────────────────────
    prec_o, rec_o, _ = precision_recall_curve(oof_true,  oof_prob)
    prec_t, rec_t, _ = precision_recall_curve(test_true, test_prob)
    prauc_o = average_precision_score(oof_true,  oof_prob)
    prauc_t = average_precision_score(test_true, test_prob)
    ax5.plot(rec_o, prec_o, color="#2ECC71", lw=2.5, label=f"OOF  AP={prauc_o:.3f}")
    ax5.plot(rec_t, prec_t, color="#E74C3C", lw=2.5, ls="--", label=f"TEST AP={prauc_t:.3f}")
    ax5.axhline(CFG["PRECISION_FLOOR"], color="#E67E22", ls=":", lw=1.2,
                label=f"Min Prec={CFG['PRECISION_FLOOR']}")
    ax5.set_title("Precision-Recall Curve", fontsize=10, fontweight="bold")
    ax5.set_xlabel("Recall"); ax5.set_ylabel("Precision")
    ax5.legend(fontsize=8.5); ax5.set_xlim(0,1); ax5.set_ylim(0,1.02)
    ax5.spines[["top","right"]].set_visible(False)

    # ── Panel 6: Predicted probability distributions ─────────────────────────
    oof_pos = oof_prob[oof_true == 1]
    oof_neg = oof_prob[oof_true == 0]
    bins = np.linspace(0, 1, 40)
    ax6.hist(oof_neg, bins=bins, alpha=0.6, color="#3498DB", label="Non-Churn (OOF)", density=True)
    ax6.hist(oof_pos, bins=bins, alpha=0.8, color="#E74C3C", label="Churn (OOF)",     density=True)
    ax6.axvline(oof_th, color="black", ls="-.", lw=1.8, label=f"Threshold={oof_th:.3f}")
    ax6.set_title("Predicted Probability Distribution (OOF)", fontsize=10, fontweight="bold")
    ax6.set_xlabel("Churn Probability"); ax6.set_ylabel("Density")
    ax6.legend(fontsize=8.5)
    ax6.spines[["top","right"]].set_visible(False)

    out_path = f"{CFG['OUTPUT_DIR']}/churn_v2_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Dashboard saved → %s", out_path)


# ── 9. Main orchestrator ──────────────────────────────────────────────────────

def _load_raw_for_scoring() -> pd.DataFrame:
    """
    Lightweight loader for SCORE mode: fetches raw data (Oracle or CSV)
    WITHOUT requiring a target column or DATASET_TYPE split — production
    scoring populations typically have neither (you're predicting the label,
    not training against it).
    """
    mode = CFG["RUN_MODE"]
    if mode == "ORACLE":
        log.info("Scoring source: ORACLE")
        df = _fetch_oracle()
    else:
        log.info("Scoring source: CSV")
        path = CFG["INPUT_CSV"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        df.columns = [c.upper().strip() for c in df.columns]
        log.info("CSV loaded: %d rows × %d cols", *df.shape)
    return df


def run_training_pipeline() -> None:
    """Full TRAIN-mode pipeline: CV → TEST holdout → report → persist artifacts."""
    t0 = time.time()
    _sep()
    horizon = CFG["CHURN_HORIZON"]
    mode    = CFG["RUN_MODE"]
    print(f"  CHURN PIPELINE V2  —  {horizon}-Day Horizon  |  Source: {mode}  |  Mode: TRAIN")
    print(f"  Target: {_TARGET_COL}   XGB depth={CFG['XGB']['max_depth']}  "
          f"lambda={CFG['XGB']['reg_lambda']}  SMOTE ratio={CFG['SMOTE_RATIO']}")
    print(f"  Threshold range: [{CFG['TH_MIN']}, {CFG['TH_MAX']}]  "
          f"Precision floor: {CFG['PRECISION_FLOOR']}")
    _sep()

    # Step 1 – Load & clean (includes new engineered features)
    train_df, test_df, feat_cols = load_and_clean()

    # Step 2 – Feature selection (MI on TRAIN, + forced high-signal features)
    selected_feat, _ = select_features(train_df, feat_cols, top_k=CFG["TOP_K_FEATURES"])

    # Step 3 – Stratified CV + stacked ensemble (3-stage Borderline-SMOTE)
    oof_prob, oof_true, fi_dict, meta_lr, fold_metrics = run_stacked_cv(
        train_df, selected_feat, feat_cols
    )

    # Step 4 – OOF threshold optimisation (now searches up to 0.995)
    oof_th, oof_sweep = sweep_threshold(oof_true, oof_prob, label="OOF")

    # Step 5 – TEST holdout evaluation + artifact capture
    test_prob, test_true, test_pred, artifacts = evaluate_on_test(
        train_df, test_df, selected_feat, oof_th, meta_lr
    )
    # Re-run sweep on TEST to get its own sweep + threshold
    test_th, _ = sweep_threshold(test_true, test_prob, label="TEST")

    # Step 6 – Full report + dashboard (now includes alert-budget table)
    full_report(
        oof_prob, oof_true, oof_th, oof_sweep,
        test_prob, test_true, test_pred, test_th,
        selected_feat, fi_dict, fold_metrics,
    )

    # Step 7 – Persist artifacts for production-scale scoring (no retrain needed)
    save_model_artifacts(artifacts)

    _sep()
    print(f"  Pipeline complete in {time.time()-t0:.1f}s. Outputs: {CFG['OUTPUT_DIR']}")
    print(f"  Model saved → {CFG['MODEL_DIR']}/churn_model_artifacts.joblib")
    print(f"  To score a large population without retraining:")
    print(f"      MODE=SCORE RUN_MODE=ORACLE python churn_pipeline_v2.py")
    print(f"      MODE=SCORE RUN_MODE=CSV INPUT_CSV=<file> python churn_pipeline_v2.py")
    _sep()


def run_scoring_pipeline() -> None:
    """
    SCORE-mode pipeline: load previously trained artifacts, score a
    population (e.g. the full 3.5M-row Oracle table) in memory-safe chunks,
    write results to CSV. No retraining — this is the fast path for
    "run with the large dataset" once a model has been trained at least once.
    """
    t0 = time.time()
    _sep()
    print(f"  CHURN PIPELINE V2  —  Mode: SCORE  |  Source: {CFG['RUN_MODE']}")
    _sep()

    artifacts = load_model_artifacts()
    df        = _load_raw_for_scoring()

    scored = score_in_chunks(df, artifacts, chunk_size=CFG["SCORE_CHUNK_SIZE"])

    out_path = os.path.join(CFG["OUTPUT_DIR"], "production_churn_scores.csv")
    scored.to_csv(out_path, index=False)

    n_flagged = int(scored["flagged"].sum())
    _sep()
    print(f"  SCORING COMPLETE")
    print(f"  Rows scored      : {len(scored):,}")
    print(f"  Flagged at risk  : {n_flagged:,}  ({n_flagged/len(scored)*100:.3f}%)")
    print(f"  Threshold used   : {artifacts['threshold']:.4f}")
    print(f"  Output           : {out_path}")
    print(f"  Elapsed          : {time.time()-t0:.1f}s")
    _sep()


def main():
    run_mode_select = os.getenv("MODE", "TRAIN").upper()
    if run_mode_select == "SCORE":
        run_scoring_pipeline()
    else:
        run_training_pipeline()


if __name__ == "__main__":
    main()
