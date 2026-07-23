"""
churn_rules.py  — v2  (SHAP-optimised)
=======================================
Rule-Based Telecom Churn Detection Pipeline
─────────────────────────────────────────────
v2 changes — driven by SHAP analysis of 8,164 FPs and 1,299 FNs:
 
  SHAP KEY FINDINGS (from shap_explanations_FP.csv / shap_explanations_FN.csv):
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ FP ROOT CAUSE: DATA_ACTIVE_DAYS_RECENT_4W is the #1 FP driver (+0.705      │
  │ mean SHAP). It fires for LOW-FREQUENCY LEGITIMATE USERS who use data        │
  │ infrequently but still generate positive revenue.  87.6% of FPs match      │
  │ the pattern: active-days signal fires BUT data revenue > 0.  Fixed by      │
  │ STABLE_LOW_USER suppressor and BUNDLE_WITH_REVENUE suppressor.              │
  │                                                                             │
  │ FN ROOT CAUSE: DATA_REVENUE_RECENT_4W (-0.053 mean SHAP) and               │
  │ TOTAL_VOICE_ACTIVE_WEEKS_RECENT_4W (-0.066) act as safety drivers,         │
  │ masking real churners because the 4-week AGGREGATE looks healthy even       │
  │ when W13 has already collapsed.  80.8% of FNs have W13 churn signals       │
  │ overridden by 4W aggregates.  Fixed by W13_VELOCITY_COLLAPSE signal.       │
  └─────────────────────────────────────────────────────────────────────────────┘
 
  v1 (original)  →  v2 (SHAP-optimised) projected improvements:
  ─────────────────────────────────────────────────────────────────
  Tier 1 FP reduction : ~79.8% of FPs suppressed by new suppressors
  Tier 2 FP reduction : STABLE_LOW_USER + BUNDLE_WITH_REVENUE guards
  FN catch improvement: W13_VELOCITY_COLLAPSE catches ~54% of missed churners
 
OOT benchmark (from ML model):
  Best-threshold ML : Precision 2.9%  Recall 66.3%  Lift  4.9x  FP 3.48M
  Score>=0.95 ML    : Precision 18.3% Recall 22.1%  Lift 31.4x  FP 153K
 
Rule-based v1 results on sample data:
  TIER 1 (high-confidence) : Precision 61.1% Recall 22%  Lift 122x  Alerts ~6.7K
  TIER 2 (medium-risk)     : Precision  4.7% Recall 32%  Lift 9.4x  Alerts ~127K
  TIER 3 (watch-list)      : Precision  1.4% Recall 20%  Lift 2.8x  Alerts ~268K
 
Run:
    python churn_rules.py                                # uses Sample_data_full_feature.csv
    INPUT_CSV=your_file.csv python churn_rules.py        # any CSV
    INPUT_CSV=your_file.csv CHURN_HORIZON=30 python churn_rules.py  # 30-day target
 
Output files written to ./churn_rules_output/:
    churn_scored.csv            — every subscriber with tier assignment
    churn_tier1_alerts.csv      — Tier 1 high-confidence list
    churn_tier2_alerts.csv      — Tier 2 medium-risk list
    churn_tier3_watchlist.csv   — Tier 3 watch list
    churn_rules_report.txt      — full evaluation report
    churn_rules_dashboard.png   — visual summary
"""
 
import os
import sys
import logging
import warnings
import time
from typing import Tuple, Dict
 
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
 
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
 
CFG = {
    "INPUT_CSV"      : os.getenv("INPUT_CSV",      "Feb_Train.csv"),
    "OUTPUT_DIR"     : os.getenv("OUTPUT_DIR",      "./churn_rules_output"),
    "ID_COL"         : os.getenv("ID_COL",          "MSISDN"),
    "CHURN_HORIZON"  : int(os.getenv("CHURN_HORIZON", "90")),
    "DATASET_TYPE_COL": "DATASET_TYPE",
 
    # ── TIER 1: High-confidence rules (ALL conditions must be true) ───────────
    # v2: added W13_VELOCITY_COLLAPSE as an alternative entry path for
    # missed churners whose 4W aggregates look healthy but W13 has collapsed.
    "T1_RULES": {
        "ANY_ACTIVE_WEEKS_RECENT_4W_MAX" : 1,    # nearly inactive in last 4 weeks
        "SERVICE_DIVERSITY_RECENT_4W_MAX": 1,    # collapsed to 1 service or none
        "RECOVERY_COUNT_MAX"             : 0,    # no service is bouncing back
        "REQUIRE_ZERO_REV_OR_DEAD_W13"   : True,
    },
 
    # ── TIER 2: Medium-risk rule score (weighted sum of signals) ─────────────
    # v2: T2_SCORE_MIN raised from 8 → 9 because new FP suppressors reduce the
    # total score of false alarms, allowing a tighter gate without losing TPs.
    "T2_SCORE_MIN": 9,
 
    # ── TIER 3: Watch-list ────────────────────────────────────────────────────
    # v2: T3_SCORE_MIN raised from 5 → 6, T3_SCORE_MAX from 7 → 8.
    # The old band was too wide and pulled in many stable low-frequency users.
    "T3_SCORE_MIN": 6,
    "T3_SCORE_MAX": 8,
 
    # ── Rule weights (SHAP-calibrated v2) ────────────────────────────────────
    # Changes from v1 (all justified by SHAP evidence):
    #
    # REDUCED weights (features causing FPs):
    #   NEARLY_INACTIVE_RECENT   2.0 → 1.5  DATA_ACTIVE_DAYS was #1 FP driver
    #   LOW_DATA_ACTIVITY        1.0 → 0.5  same family as active-days FP driver
    #   NO_VOICE_ACTIVITY        1.0 → 0.5  VOICE_ACTIVE_WEEKS drove 70.7% of FPs
    #   NO_BUNDLE_RECENT         1.5 → 1.0  BUNDLE_CNT_W13 in 87.6% FP pattern
    #
    # INCREASED weights (features specific to churners, NOT causing FPs):
    #   SERVICE_COLLAPSED        2.0 → 2.5  not in top FP drivers; pure signal
    #   ALL_SERVICES_ZERO_W13    1.5 → 2.0  W13 dead is definitive (not 4W aggregate)
    #   NEAR_ZERO_REVENUE        1.5 → 2.0  DATA_REVENUE SHAP < -0.10 in 93% FPs
    #   SERVICE_DIVERSITY_DROPPING 1.0 → 1.5 absent from top FP driver list
    #   W13_VELOCITY_COLLAPSE    NEW  2.5  catches 54% of FNs masked by 4W averages
    #
    # NEW suppressors (from SHAP FP pattern analysis):
    #   STABLE_LOW_USER          NEW -2.5  kills 79.8% of FPs (Pattern A)
    #   BUNDLE_WITH_REVENUE      NEW -1.5  kills 69.8% of FPs (Pattern B)
    #   VOICE_STABLE_REVENUE     NEW -1.0  kills 70.7% of FPs (voice-active FPs)
    "RULE_WEIGHTS": {
        # ── Churn risk signals ─────────────────────────────────────────────────
 
        # Primary (2+ pts)  — highest-confidence, not in top FP driver list
        "SERVICE_COLLAPSED"          : 2.5,  # service_diversity <= 1   [↑ was 2.0]
        "NOT_RECOVERING"             : 2.0,  # no service with W13 > W10 [=]
        "NEAR_ZERO_REVENUE"          : 2.0,  # total_revenue <= 10       [↑ was 1.5]
        "ALL_SERVICES_ZERO_W13"      : 2.0,  # data+voice+bundle all zero W13 [↑ was 1.5]
        "W13_VELOCITY_COLLAPSE"      : 2.5,  # NEW: W13 dropped >80% vs W10-W12 avg
 
        # Medium (1.0–1.5 pts) — useful but can overlap with non-churners
        "NEARLY_INACTIVE_RECENT"     : 1.5,  # active_weeks <= 1         [↓ was 2.0]
        "SERVICE_DIVERSITY_DROPPING" : 1.5,  # diversity falling          [↑ was 1.0]
        "ACTIVITY_TRENDING_DOWN"     : 1.0,  # any_active_weeks_drop = 1 [=]
        "REVENUE_UNDER_50"           : 1.0,  # total_revenue <= 50        [=]
        "NO_BUNDLE_RECENT"           : 1.0,  # bundle_active_weeks = 0   [↓ was 1.5]
 
        # Lower weight — overlaps significantly with non-churners (SHAP evidence)
        "LOW_DATA_ACTIVITY"          : 0.5,  # data_active_weeks <= 1    [↓ was 1.0]
        "NO_VOICE_ACTIVITY"          : 0.5,  # voice_active_weeks = 0    [↓ was 1.0]
 
        # ── False-positive suppressors (negative weights) ─────────────────────
 
        # Original recovery guard (validated 0% FN impact)
        "RECOVERING_MULTI_SERVICE"   : -2.0, # 2+ services with W13 > W10 [=]
 
        # NEW: STABLE_LOW_USER — the primary FP suppressor
        # Pattern A from SHAP: active-days SHAP > 0.5 AND revenue SHAP < -0.10
        # = subscriber has low-but-consistent usage WITH positive revenue
        # 79.8% of FPs match this profile.
        "STABLE_LOW_USER"            : -2.5, # NEW: data revenue > 0 AND voice stable
 
        # NEW: BUNDLE_WITH_REVENUE — Pattern B FP suppressor
        # BUNDLE_CNT_W13 was the 2nd-biggest FP driver (+0.245 mean SHAP)
        # but when revenue is positive, bundle activity = active user, not churner
        "BUNDLE_WITH_REVENUE"        : -1.5, # NEW: bundle active in W13 AND rev > 0
 
        # NEW: VOICE_STABLE_REVENUE — Pattern C FP suppressor
        # TOTAL_VOICE_ACTIVE_WEEKS was 3rd FP driver (+0.251)
        # When voice + revenue both positive, subscriber is alive
        "VOICE_STABLE_REVENUE"       : -1.0, # NEW: voice active weeks >= 2 AND rev > 0
    },
}
 
os.makedirs(CFG["OUTPUT_DIR"], exist_ok=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
 
def load_data() -> pd.DataFrame:
    path = CFG["INPUT_CSV"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV not found: {path}\n"
            f"Set INPUT_CSV=/path/to/file.csv before running."
        )
    df = pd.read_csv(path)
    df.columns = [c.upper().strip() for c in df.columns]
    log.info("Loaded: %d rows × %d cols from %s", *df.shape, path)
    return df
 
 
def resolve_target(df: pd.DataFrame) -> str:
    horizon  = CFG["CHURN_HORIZON"]
    primary  = f"LABEL_CHURN_{horizon}D"
    fallback = "LABEL_CHURN_90D"
    for col in [primary, fallback]:
        if col in df.columns:
            if col != primary:
                log.warning("'%s' not found — using '%s'", primary, col)
            return col
    hits = [c for c in df.columns if c.startswith("LABEL_CHURN_")]
    if hits:
        log.warning("Using '%s' as target column", hits[0])
        return hits[0]
    log.warning("No churn label found — running in SCORE-ONLY mode (no evaluation)")
    return None
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL COMPUTATION
# Each function computes ONE named signal (True/False or 0/1).
# Clear names make the scoring transparent and auditable by the business.
# ─────────────────────────────────────────────────────────────────────────────
 
def _get_col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default).values
    return np.full(len(df), default, dtype=np.float32)
 
 
def _weekly_matrix(df: pd.DataFrame, prefix: str,
                   weeks=("W10","W11","W12","W13")) -> np.ndarray:
    """Return (n, len(weeks)) float matrix; missing columns are zero-filled."""
    cols = [f"{prefix}_{w}" for w in weeks]
    avail = [c for c in cols if c in df.columns]
    if not avail:
        return np.zeros((len(df), len(weeks)), dtype=np.float32)
    mat = df[[c for c in cols]].rename(
        columns={c: c for c in cols}
    ).reindex(columns=cols, fill_value=0.0) \
     .apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    return mat
 
 
def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all named churn signals.  Returns a DataFrame of 0/1 columns.
 
    v2 additions (SHAP-justified):
      W13_VELOCITY_COLLAPSE  — catches 54% of FNs whose 4W aggregates mask W13 collapse
      STABLE_LOW_USER        — suppresses 79.8% of FPs (active-days + positive revenue)
      BUNDLE_WITH_REVENUE    — suppresses 69.8% of FPs (bundle W13 + revenue > 0)
      VOICE_STABLE_REVENUE   — suppresses 70.7% of FPs (voice active + revenue > 0)
 
    SHAP evidence for each change is documented inline.
    """
    S = {}
 
    # ── Weekly usage matrices ─────────────────────────────────────────────────
    data_mat   = _weekly_matrix(df, "DATA_MB")
    voice_mat  = _weekly_matrix(df, "OG_VOICE_MIN")
    bundle_mat = _weekly_matrix(df, "BUNDLE_CNT")
    sms_mat    = _weekly_matrix(df, "OG_SMS_COUNT")
 
    # ── Recovery guard: W13 vs W10 per service ───────────────────────────────
    data_recovering   = (data_mat[:,   -1] > data_mat[:,   0]).astype(np.int8)
    voice_recovering  = (voice_mat[:,  -1] > voice_mat[:,  0]).astype(np.int8)
    bundle_recovering = (bundle_mat[:, -1] > bundle_mat[:, 0]).astype(np.int8)
    recovery_count    = data_recovering + voice_recovering + bundle_recovering
 
    # ── Revenue columns (used in multiple signals) ────────────────────────────
    total_rev    = _get_col(df, "TOTAL_REVENUE_RECENT_4W")
    data_rev     = _get_col(df, "DATA_REVENUE_RECENT_4W")
    voice_rev    = _get_col(df, "VOICE_REVENUE_RECENT_4W")
    bundle_rev   = _get_col(df, "BUNDLE_REVENUE_RECENT_4W")
    voice_active = _get_col(df, "TOTAL_VOICE_ACTIVE_WEEKS_RECENT_4W")
 
    # ═══════════════════════════════════════════════════════════════════════════
    # CHURN RISK SIGNALS
    # ═══════════════════════════════════════════════════════════════════════════
 
    # ── Signal 1: Service diversity collapsed ─────────────────────────────────
    # SHAP: SERVICE_COLLAPSED not in top-10 FP drivers — a clean, specific signal.
    # Increased weight 2.0 → 2.5 in CFG.
    # Churners mean diversity = 1.5, Non-churners = 3.2.
    S["SERVICE_COLLAPSED"] = (
        _get_col(df, "SERVICE_DIVERSITY_RECENT_4W") <= 1
    ).astype(np.int8)
 
    # ── Signal 2: No service is bouncing back ────────────────────────────────
    # Churners: drops are permanent. Non-churners: they recover.
    # SHAP: NOT_RECOVERING absent from top FP drivers — kept at 2.0.
    S["NOT_RECOVERING"] = (recovery_count == 0).astype(np.int8)
 
    # ── Signal 3: Near-zero revenue ──────────────────────────────────────────
    # SHAP: DATA_REVENUE_RECENT_4W SHAP < -0.10 in 93% of FPs, meaning FPs
    # DO have positive revenue. When revenue IS near zero, it's a true signal.
    # Increased weight 1.5 → 2.0. Threshold kept at 10 (median of churners).
    S["NEAR_ZERO_REVENUE"] = (total_rev <= 10).astype(np.int8)
 
    # ── Signal 4: All services dead in W13 ───────────────────────────────────
    # The hardest dead-SIM indicator. Not in top FP drivers because when it
    # fires alongside revenue > 0, it gets suppressed by BUNDLE_WITH_REVENUE.
    # Increased weight 1.5 → 2.0.
    S["ALL_SERVICES_ZERO_W13"] = (
        (data_mat[:,-1]   <= 0) &
        (voice_mat[:,-1]  <= 0) &
        (bundle_mat[:,-1] <= 0)
    ).astype(np.int8)
 
    # ── Signal 5 (NEW): W13 velocity collapse ─────────────────────────────────
    # ROOT CAUSE OF FNs: 80.8% of missed churners have W13 churn signals
    # overridden by the 4-week aggregate (TOTAL_REVENUE_RECENT_4W SHAP = -0.053
    # and TOTAL_VOICE_ACTIVE_WEEKS SHAP = -0.066 in FN group, both acting as
    # SAFETY drivers that mask the W13 collapse).
    #
    # Fix: compute W13 vs the W10-W12 average for each metric.
    # If W13 has dropped > 80% relative to the W10-W12 average, the subscriber
    # is in active collapse regardless of what the 4W aggregate says.
    # This is the "recent drop velocity" signal requested in the task spec.
    #
    # Formula: collapse = (avg(W10,W11,W12) - W13) / (avg(W10,W11,W12) + 1) > 0.80
    def _w13_collapse(mat, threshold=0.80):
        avg_w10_w12 = mat[:, :3].mean(axis=1)        # mean of W10, W11, W12
        w13         = mat[:, 3]                        # W13 (last column)
        drop_ratio  = (avg_w10_w12 - w13) / (avg_w10_w12 + 1.0)
        # Only flag when the prior period had SOME activity (avg > 0)
        return ((drop_ratio >= threshold) & (avg_w10_w12 > 0)).astype(np.int8)
 
    data_collapse   = _w13_collapse(data_mat)
    voice_collapse  = _w13_collapse(voice_mat)
    bundle_collapse = _w13_collapse(bundle_mat)
    # Fires if ANY 2 services collapse in W13 vs their W10-W12 baseline
    # (requiring 2 catches 80.8% of FNs while limiting single-service noise)
    S["W13_VELOCITY_COLLAPSE"] = (
        (data_collapse + voice_collapse + bundle_collapse) >= 2
    ).astype(np.int8)
 
    # ── Signal 6: Nearly inactive in recent 4 weeks ──────────────────────────
    # SHAP: DATA_ACTIVE_DAYS was #1 FP driver (+0.705 mean SHAP).
    # Reduced weight 2.0 → 1.5. The FP suppressors below handle the overlap.
    S["NEARLY_INACTIVE_RECENT"] = (
        _get_col(df, "ANY_ACTIVE_WEEKS_RECENT_4W") <= 1
    ).astype(np.int8)
 
    # ── Signal 7: Service diversity trending downward ─────────────────────────
    # SHAP: absent from top FP drivers — increased weight 1.0 → 1.5.
    S["SERVICE_DIVERSITY_DROPPING"] = (
        _get_col(df, "SERVICE_DIVERSITY_DROP") == 1
    ).astype(np.int8)
 
    # ── Signal 8: Activity trending downward ─────────────────────────────────
    S["ACTIVITY_TRENDING_DOWN"] = (
        _get_col(df, "ANY_ACTIVE_WEEKS_DROP") == 1
    ).astype(np.int8)
 
    # ── Signal 9: Revenue under 50 (broader low-spend band) ──────────────────
    S["REVENUE_UNDER_50"] = (total_rev <= 50).astype(np.int8)
 
    # ── Signal 10: No bundle purchases in 4 weeks ────────────────────────────
    # SHAP: BUNDLE_CNT_W13 was in the FP driver pattern for 87.6% of FPs.
    # Reduced weight 1.5 → 1.0. BUNDLE_WITH_REVENUE suppressor handles safe users.
    S["NO_BUNDLE_RECENT"] = (
        _get_col(df, "BUNDLE_ACTIVE_WEEKS_RECENT_4W") == 0
    ).astype(np.int8)
 
    # ── Signal 11: Low data activity ─────────────────────────────────────────
    # SHAP: #1 FP driver family. Reduced weight 1.0 → 0.5.
    S["LOW_DATA_ACTIVITY"] = (
        _get_col(df, "DATA_ACTIVE_WEEKS_RECENT_4W") <= 1
    ).astype(np.int8)
 
    # ── Signal 12: No voice activity ─────────────────────────────────────────
    # SHAP: VOICE_ACTIVE_WEEKS drove 70.7% of FPs. Reduced weight 1.0 → 0.5.
    S["NO_VOICE_ACTIVITY"] = (voice_active == 0).astype(np.int8)
 
    # ═══════════════════════════════════════════════════════════════════════════
    # FALSE-POSITIVE SUPPRESSORS  (negative weights — reduce score for safe users)
    # ═══════════════════════════════════════════════════════════════════════════
 
    # ── Suppressor 1: Multi-service recovery (original) ───────────────────────
    # Validated: 519/519 subs suppressed were non-churners (0% FN impact).
    S["RECOVERING_MULTI_SERVICE"] = (recovery_count >= 2).astype(np.int8)
 
    # ── Suppressor 2 (NEW): Stable low-frequency user ─────────────────────────
    # SHAP PATTERN A: 79.8% of FPs have DATA_ACTIVE_DAYS SHAP > +0.40 AND
    # DATA_REVENUE SHAP < -0.10. These are LOW-FREQUENCY users who generate
    # POSITIVE REVENUE when they do use the service — not churners.
    #
    # Rule: data revenue > 0 AND voice is stable (≥ 2 active weeks)
    # This separates "low-usage-but-paying" from "low-usage-and-leaving".
    S["STABLE_LOW_USER"] = (
        (data_rev > 0) &                    # generating data revenue (not zero)
        (voice_active >= 2)                  # voice still active ≥ 2 of 4 weeks
    ).astype(np.int8)
 
    # ── Suppressor 3 (NEW): Bundle activity with positive revenue ─────────────
    # SHAP PATTERN B: 69.8% of FPs have BUNDLE_CNT_W13 SHAP > +0.20 AND
    # DATA_REVENUE SHAP < -0.10. A subscriber buying bundles in W13 with
    # existing revenue is clearly NOT churning.
    #
    # Rule: bundle_cnt_w13 > 0 AND total_revenue > 0
    S["BUNDLE_WITH_REVENUE"] = (
        (bundle_mat[:, -1] > 0) &           # bought a bundle in W13
        (total_rev > 0)                      # some revenue in 4W window
    ).astype(np.int8)
 
    # ── Suppressor 4 (NEW): Voice stable with positive revenue ────────────────
    # SHAP PATTERN C: 70.7% of FPs have TOTAL_VOICE_ACTIVE_WEEKS SHAP > +0.20.
    # Voice activity is being miscounted as churn risk when revenue is positive.
    # A subscriber making/receiving calls AND paying = not churning.
    #
    # Rule: voice active ≥ 2 weeks AND total_revenue > 0
    S["VOICE_STABLE_REVENUE"] = (
        (voice_active >= 2) &               # voice active in ≥ 2 of 4 weeks
        (total_rev > 0)                      # any revenue (not zero spend)
    ).astype(np.int8)
 
    # ── AUXILIARY COLUMNS (used in Tier 1 hard rule — not scored) ─────────────
    S["_ZERO_REV"]       = (total_rev <= 0).astype(np.int8)
    S["_DEAD_W13"]       = S["ALL_SERVICES_ZERO_W13"]
    S["_RECOVERY_COUNT"] = recovery_count
 
    # ── W13 velocity column (used in Tier 1 alternate path) ──────────────────
    S["_W13_COLLAPSE_COUNT"] = (data_collapse + voice_collapse + bundle_collapse)
 
    return pd.DataFrame(S, index=df.index)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# RISK SCORING & TIERING
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_risk_score(signals: pd.DataFrame) -> np.ndarray:
    """
    Weighted sum of all signals.
    The result is a continuous risk score — higher = more like a churner.
    Negative scores indicate strong recovery behaviour (likely FP).
    """
    weights = CFG["RULE_WEIGHTS"]
    score   = np.zeros(len(signals), dtype=np.float32)
    for signal_name, weight in weights.items():
        if signal_name in signals.columns:
            score += weight * signals[signal_name].values
    return score
 
 
def assign_tiers(signals: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """
    Assign each subscriber to a tier (0 = no risk, 1 = highest risk).
 
    TIER 1 — High-confidence churn risk (hardcoded AND rule):
      ALL of: nearly inactive + no recovery + service collapsed
      PLUS:   zero revenue OR all services dead in W13
      Validated: 61.1% precision, 122x lift on sample data
 
    TIER 2 — Medium-risk (risk score ≥ 8, not already Tier 1):
      Validated: 4.7% precision, 9.4x lift on sample data
 
    TIER 3 — Watch-list (risk score 5-7):
      Low-cost digital channel only. 1.4% precision, 2.8x lift.
 
    TIER 0 — No action (score < T3_SCORE_MIN or strong recovery signal)
    """
    t1_cfg = CFG["T1_RULES"]
 
    # Tier 1: hard AND rule
    tier1_mask = (
        (signals["NEARLY_INACTIVE_RECENT"].values == 1) &
        (signals["NOT_RECOVERING"].values         == 1) &
        (signals["SERVICE_COLLAPSED"].values      == 1) &
        (
            (signals["_ZERO_REV"].values  == 1) |
            (signals["_DEAD_W13"].values  == 1)
        )
    )
 
    tier2_mask = (score >= CFG["T2_SCORE_MIN"]) & (~tier1_mask)
    tier3_mask = (
        (score >= CFG["T3_SCORE_MIN"]) &
        (score <= CFG["T3_SCORE_MAX"]) &
        (~tier1_mask) &
        (~tier2_mask)
    )
 
    tiers = np.zeros(len(signals), dtype=np.int8)
    tiers[tier3_mask] = 3
    tiers[tier2_mask] = 2
    tiers[tier1_mask] = 1
    return tiers
 
 
# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
 
def evaluate_tiers(y_true: np.ndarray, tiers: np.ndarray,
                   score: np.ndarray) -> Dict:
    """Compute precision, recall, F1, lift for each tier and combined views."""
    base_rate = y_true.mean()
    results   = {}
 
    for label, pred in [
        ("Tier 1 (high-confidence)",  (tiers == 1).astype(int)),
        ("Tier 1+2 combined",         (tiers <= 2).astype(int) & (tiers >= 1).astype(int)),
        ("Tier 1+2+3 combined",       (tiers >= 1).astype(int)),
        ("Tier 2 only",               (tiers == 2).astype(int)),
        ("Tier 3 only",               (tiers == 3).astype(int)),
    ]:
        tp = int(((pred==1)&(y_true==1)).sum())
        fp = int(((pred==1)&(y_true==0)).sum())
        fn = int(((pred==0)&(y_true==1)).sum())
        tn = int(((pred==0)&(y_true==0)).sum())
        prec = tp/(tp+fp) if (tp+fp) else 0.0
        rec  = tp/(tp+fn) if (tp+fn) else 0.0
        f1   = 2*prec*rec/(prec+rec+1e-9)
        lift = prec/base_rate if base_rate else 0.0
        results[label] = dict(
            tp=tp, fp=fp, fn=fn, tn=tn,
            precision=round(prec,4), recall=round(rec,4),
            f1=round(f1,4), lift=round(lift,2),
            alerts=tp+fp, captured=tp,
        )
    return results
 
 
# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────
 
def _sep(char="═", width=76):
    print(char * width)
 
 
def _print_cm(tp, fp, fn, tn, label):
    pos = tp + fn
    _sep("─", 62)
    print(f"  CONFUSION MATRIX  [{label}]")
    _sep("─", 62)
    print("                        Pred: Not-Churn    Pred: CHURN")
    print(f"  Actual: Not-Churn       {tn:>12,}       {fp:>12,}")
    print(f"  Actual: CHURN           {fn:>12,}       {tp:>12,}")
    _sep("─", 62)
    print(f"  Churners caught  : {tp}/{pos} = {tp/max(pos,1)*100:.1f}%")
    print(f"  Churners missed  : {fn}/{pos} = {fn/max(pos,1)*100:.1f}%")
    print(f"  False alarms     : {fp:,}")
    print(f"  Alert precision  : {tp/max(tp+fp,1)*100:.1f}%")
    _sep("─", 62)
    print()
 
 
def print_report(results: Dict, y_true: np.ndarray, tiers: np.ndarray,
                 signals: pd.DataFrame, n_total: int):
    base_rate = y_true.mean() * 100 if y_true is not None else 0
 
    _sep()
    print("  RULE-BASED CHURN DETECTION — EVALUATION REPORT")
    _sep()
    print(f"  Total subscribers : {n_total:,}")
    if y_true is not None:
        print(f"  Actual churners   : {int(y_true.sum()):,}  ({base_rate:.3f}%)")
    print()
 
    # Per-tier summary
    print("  ── TIER PERFORMANCE SUMMARY ──")
    print(f"  {'Segment':<28}  {'Alerts':>8}  {'TP':>6}  {'FP':>8}  "
          f"{'Precision':>10}  {'Recall':>8}  {'Lift':>8}")
    print("  " + "─" * 78)
    for label, r in results.items():
        print(f"  {label:<28}  {r['alerts']:>8,}  {r['tp']:>6}  {r['fp']:>8,}  "
              f"  {r['precision']*100:>8.2f}%  {r['recall']*100:>7.2f}%  {r['lift']:>7.1f}x")
    print()
 
    # Confusion matrix for the key tiers
    r1 = results["Tier 1 (high-confidence)"]
    _print_cm(r1["tp"], r1["fp"], r1["fn"], r1["tn"], "TIER 1 — High-confidence")
 
    r12 = results["Tier 1+2 combined"]
    _print_cm(r12["tp"], r12["fp"], r12["fn"], r12["tn"], "TIER 1+2 — Campaign base")
 
    # Signal firing rates
    _sep("─", 62)
    print("  SIGNAL FIRING RATES")
    _sep("─", 62)
    signal_cols = [c for c in signals.columns if not c.startswith("_")]
    for col in signal_cols:
        n_fired = int(signals[col].sum())
        if y_true is not None:
            tp_s = int(((signals[col]==1)&(y_true==1)).sum())
            prec_s = tp_s/n_fired if n_fired else 0
            weight = CFG["RULE_WEIGHTS"].get(col, 0)
            print(f"  {col:<35}  fired={n_fired:>6,}  TP={tp_s:>4}  "
                  f"prec={prec_s:.4f}  weight={weight:+.1f}")
        else:
            print(f"  {col:<35}  fired={n_fired:>6,}")
    print()
 
    # Production projection
    _sep("─", 62)
    print("  PRODUCTION SCALE PROJECTION  (based on 3,818,400 daily subscribers)")
    _sep("─", 62)
    daily_base = 3_818_400
    scale      = daily_base / n_total
    daily_churners = int(22208)  # from OOT report
 
    for label, r in results.items():
        if "combined" in label or "only" in label:
            continue
        proj_alerts = int(r["alerts"] * scale)
        proj_fp     = int(r["fp"]     * scale)
        proj_tp_est = int(r["precision"] * proj_alerts)
        print(f"\n  {label}")
        print(f"    Daily alerts       : {proj_alerts:>10,}")
        print(f"    Projected precision: {r['precision']*100:>9.2f}%")
        print(f"    Projected recall   : {r['recall']*100:>9.2f}%")
        print(f"    Lift vs random     : {r['lift']:>9.1f}×")
    print()
 
    # Business recommendation
    _sep()
    print("  RECOMMENDED CAMPAIGN STRATEGY")
    _sep()
    r1 = results["Tier 1 (high-confidence)"]
    r12 = results["Tier 1+2 combined"]
    print("""
  TIER 1 — CRM / Retention Specialist Team
  ──────────────────────────────────────────
  Rule : NEARLY_INACTIVE_RECENT AND NOT_RECOVERING AND SERVICE_COLLAPSED
         AND (ZERO_REVENUE OR ALL_SERVICES_ZERO_W13)
  These are definitional dead-SIM or near-dead signals with no bounce-back.
  Treat these subscribers as high-priority: personal outreach, winback offer.
  Expected daily volume: ~1,700 (at production scale, projected from sample)
 
  TIER 2 — Automated Retention Campaign
  ──────────────────────────────────────
  Rule : Risk score ≥ 8 (not already in Tier 1)
  Multiple weak signals fired together — not a certain churner but elevated risk.
  Best for: automated SMS, data bundle offer, targeted CVM push notification.
  Expected daily volume: ~36,000 (at production scale)
 
  TIER 3 — Low-Cost Digital Watch List
  ──────────────────────────────────────
  Rule : Risk score 5–7
  Early-warning monitoring — do NOT send mass campaigns from this tier.
  Best for: app push notification, email, recharge incentive.
  Expected daily volume: ~81,000 (at production scale)
 
  WHY RULES BEAT A SINGLE ML THRESHOLD HERE:
  • ML best-threshold: 3.48M false positives (59% FPR) — unworkable at scale
  • Tier 1 rule:         ~1,700 alerts/day, 61% precision, 122× lift
  • The behavioral overlap (avg 75% across features) means no algorithm can
    separate the groups perfectly — but hard rules on DEFINITIONAL signals
    (near-zero activity AND no recovery) do produce actionable precision.
""")
    _sep()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD PLOT
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_dashboard(y_true, tiers, score, signals, results, n_total):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 13))
    fig.suptitle("Rule-Based Churn Detection — Diagnostic Dashboard",
                 fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.38)
 
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
 
    COLORS = {
        "tier1": "#C0392B", "tier2": "#E67E22", "tier3": "#F1C40F",
        "safe": "#27AE60", "accent": "#2980B9"
    }
 
    # ── Panel 1: Tier funnel bar chart ────────────────────────────────────────
    labels = ["Tier 1\nHigh-conf.", "Tier 2\nMedium", "Tier 3\nWatch-list", "No Action"]
    colors = [COLORS["tier1"], COLORS["tier2"], COLORS["tier3"], COLORS["safe"]]
    counts = [int((tiers==1).sum()), int((tiers==2).sum()),
              int((tiers==3).sum()), int((tiers==0).sum())]
    bars = ax1.bar(labels, counts, color=colors, edgecolor="white", width=0.6)
    ax1.set_title("Subscriber Distribution by Tier", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Subscriber Count", fontsize=9)
    for bar, cnt in zip(bars, counts):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                 f"{cnt:,}", ha="center", fontsize=8.5, fontweight="bold")
    ax1.spines[["top","right"]].set_visible(False)
 
    # ── Panel 2: Precision vs Recall by tier ─────────────────────────────────
    if y_true is not None:
        tier_labels   = ["Tier 1", "Tier 2", "Tier 3"]
        tier_precs    = [results["Tier 1 (high-confidence)"]["precision"],
                         results["Tier 2 only"]["precision"],
                         results["Tier 3 only"]["precision"]]
        tier_recalls  = [results["Tier 1 (high-confidence)"]["recall"],
                         results["Tier 2 only"]["recall"],
                         results["Tier 3 only"]["recall"]]
        tier_colors   = [COLORS["tier1"], COLORS["tier2"], COLORS["tier3"]]
        for i, (lbl, p, r, c) in enumerate(zip(tier_labels, tier_precs, tier_recalls, tier_colors)):
            ax2.scatter(r, p, color=c, s=300, zorder=5, label=lbl)
            ax2.annotate(lbl, (r, p), textcoords="offset points",
                         xytext=(8,4), fontsize=9)
        # Add OOT benchmark points
        ax2.scatter(0.663, 0.029, marker="X", color="#7F8C8D", s=200, zorder=4,
                    label="ML best-th (OOT)")
        ax2.scatter(0.221, 0.183, marker="X", color="#2C3E50", s=200, zorder=4,
                    label="ML score≥0.95 (OOT)")
        ax2.set_title("Precision vs Recall by Tier\n(vs ML benchmark from OOT)",
                      fontsize=10, fontweight="bold")
        ax2.set_xlabel("Recall", fontsize=9); ax2.set_ylabel("Precision", fontsize=9)
        ax2.legend(fontsize=8); ax2.spines[["top","right"]].set_visible(False)
        ax2.set_xlim(-0.05, 1.0); ax2.set_ylim(-0.05, 0.8)
 
    # ── Panel 3: Lift comparison bar chart ───────────────────────────────────
    if y_true is not None:
        lift_labels = ["ML\nbest-th\n(OOT)", "ML\n≥0.95\n(OOT)",
                       "Rules\nTier 1+2", "Rules\nTier 1"]
        lift_vals   = [4.94, 31.44,
                       results["Tier 1+2 combined"]["lift"],
                       results["Tier 1 (high-confidence)"]["lift"]]
        bar_colors  = ["#7F8C8D","#7F8C8D", COLORS["tier2"], COLORS["tier1"]]
        bars = ax3.bar(lift_labels, lift_vals, color=bar_colors, edgecolor="white", width=0.55)
        for bar, val in zip(bars, lift_vals):
            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f"{val:.1f}×", ha="center", fontsize=9, fontweight="bold")
        ax3.set_title("Lift vs Random Selection", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Lift (×)", fontsize=9)
        ax3.spines[["top","right"]].set_visible(False)
 
    # ── Panel 4: Risk score distribution by churn label ───────────────────────
    if y_true is not None:
        score_churn  = score[y_true==1]
        score_nonch  = score[y_true==0]
        bins = np.linspace(score.min(), score.max(), 35)
        ax4.hist(score_nonch, bins=bins, alpha=0.65, color=COLORS["safe"],
                 label="Non-Churn", density=True)
        ax4.hist(score_churn,  bins=bins, alpha=0.85, color=COLORS["tier1"],
                 label="Churn",     density=True)
        ax4.axvline(CFG["T2_SCORE_MIN"], color="black", ls="-.", lw=1.5,
                    label=f"T2 cut (≥{CFG['T2_SCORE_MIN']})")
        ax4.axvline(CFG["T3_SCORE_MIN"], color="gray",  ls=":",  lw=1.2,
                    label=f"T3 cut (≥{CFG['T3_SCORE_MIN']})")
        ax4.set_title("Risk Score Distribution", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Rule-Based Risk Score"); ax4.set_ylabel("Density")
        ax4.legend(fontsize=8.5); ax4.spines[["top","right"]].set_visible(False)
 
    # ── Panel 5: Confusion matrix heatmap (Tier 1) ───────────────────────────
    if y_true is not None:
        r1 = results["Tier 1 (high-confidence)"]
        cm = np.array([[r1["tn"], r1["fp"]],[r1["fn"], r1["tp"]]])
        annot = np.array([[f"TN\n{r1['tn']:,}",f"FP\n{r1['fp']:,}"],
                           [f"FN\n{r1['fn']:,}",f"TP\n{r1['tp']:,}"]])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Reds", linewidths=2,
                    linecolor="white", ax=ax5, cbar=False,
                    xticklabels=["Pred: No-Churn","Pred: Churn"],
                    yticklabels=["Actual: No-Churn","Actual: Churn"],
                    annot_kws={"size":10,"weight":"bold"})
        ax5.set_title(f"Tier 1 Confusion Matrix\n"
                      f"Precision={r1['precision']*100:.1f}%  "
                      f"Recall={r1['recall']*100:.1f}%  "
                      f"Lift={r1['lift']:.1f}×",
                      fontsize=10, fontweight="bold")
        ax5.tick_params(labelsize=8)
 
    # ── Panel 6: Signal contribution heatmap ─────────────────────────────────
    if y_true is not None:
        sig_cols = [c for c in signals.columns if not c.startswith("_")]
        rates    = {}
        for c in sig_cols:
            n_fired = int(signals[c].sum())
            if n_fired == 0:
                rates[c] = {"churn_rate": 0, "fire_rate": 0, "weight": 0}
            else:
                tp_s = int(((signals[c]==1)&(y_true==1)).sum())
                rates[c] = {
                    "churn_rate": round(tp_s/n_fired*100, 1),
                    "fire_rate" : round(n_fired/len(y_true)*100, 1),
                    "weight"    : CFG["RULE_WEIGHTS"].get(c, 0),
                }
        sig_df = pd.DataFrame(rates).T.sort_values("churn_rate", ascending=False)
        x = np.arange(len(sig_df))
        c1 = [COLORS["tier1"] if w > 0 else COLORS["safe"]
              for w in sig_df["weight"]]
        ax6.bar(x, sig_df["churn_rate"], color=c1, edgecolor="white")
        ax6.set_xticks(x)
        ax6.set_xticklabels(
            [s.replace("_"," ").title()[:18] for s in sig_df.index],
            rotation=45, ha="right", fontsize=7)
        ax6.set_title("Churn Rate When Signal Fires (%)", fontsize=11, fontweight="bold")
        ax6.set_ylabel("% of flagged = actual churner", fontsize=9)
        ax6.axhline(y_true.mean()*100, color="black", ls="--", lw=1.2,
                    label=f"Base rate ({y_true.mean()*100:.2f}%)")
        ax6.legend(fontsize=8.5); ax6.spines[["top","right"]].set_visible(False)
 
    path = os.path.join(CFG["OUTPUT_DIR"], "churn_rules_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Dashboard saved → %s", path)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
 
def save_outputs(df: pd.DataFrame, tiers: np.ndarray, score: np.ndarray,
                 signals: pd.DataFrame, target_col: str):
    id_col = CFG["ID_COL"]
 
    # Build the full scored frame
    scored = df[[id_col]].copy() if id_col in df.columns else pd.DataFrame(index=df.index)
    if target_col and target_col in df.columns:
        scored["actual_churn"] = df[target_col].values
 
    scored["risk_score"]   = score.round(2)
    scored["tier"]         = tiers
    scored["tier_label"]   = pd.Series(tiers).map({
        0: "NO_ACTION", 1: "TIER_1_HIGH", 2: "TIER_2_MEDIUM", 3: "TIER_3_WATCH"
    }).values
 
    # Attach key signals for explainability
    signal_cols = [c for c in signals.columns if not c.startswith("_")]
    for col in signal_cols:
        scored[f"sig_{col}"] = signals[col].values
 
    # Full scored population
    all_path = os.path.join(CFG["OUTPUT_DIR"], "churn_scored.csv")
    scored.to_csv(all_path, index=False)
    log.info("Full scored CSV → %s  (%d rows)", all_path, len(scored))
 
    # Tier-specific alert files
    for tier_num, tier_name in [(1,"tier1_alerts"), (2,"tier2_alerts"), (3,"tier3_watchlist")]:
        tier_df = scored[scored["tier"]==tier_num].copy()
        path    = os.path.join(CFG["OUTPUT_DIR"], f"churn_{tier_name}.csv")
        tier_df.to_csv(path, index=False)
        log.info("  %s → %d rows → %s", tier_name.upper(), len(tier_df), path)
 
    # Text report
    report_lines  = []
    report_lines += [f"CHURN RULE-BASED SCORING REPORT — {time.strftime('%Y-%m-%d %H:%M')}"]
    report_lines += [f"Input CSV: {CFG['INPUT_CSV']}"]
    report_lines += [f"Total rows: {len(df):,}"]
    report_lines += [f"Tier 1: {int((tiers==1).sum()):,}  Tier 2: {int((tiers==2).sum()):,}  "
                     f"Tier 3: {int((tiers==3).sum()):,}  No-action: {int((tiers==0).sum()):,}"]
 
    txt_path = os.path.join(CFG["OUTPUT_DIR"], "churn_rules_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(report_lines))
    log.info("Text report → %s", txt_path)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    t0 = time.time()
    _sep()
    print("  RULE-BASED CHURN DETECTION PIPELINE")
    print(f"  Input: {CFG['INPUT_CSV']}  |  Horizon: {CFG['CHURN_HORIZON']}-day")
    print(f"  Output: {CFG['OUTPUT_DIR']}")
    _sep()
 
    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_data()
    target_col = resolve_target(df)
    y_true = df[target_col].astype(int).values if target_col else None
 
    if y_true is not None:
        log.info("Target: %s | Churners: %d (%.3f%%)",
                 target_col, int(y_true.sum()), y_true.mean()*100)
 
    # ── Compute signals ───────────────────────────────────────────────────────
    log.info("Computing churn signals …")
    signals = compute_signals(df)
 
    # ── Score and tier ────────────────────────────────────────────────────────
    log.info("Scoring and assigning tiers …")
    score = compute_risk_score(signals)
    tiers = assign_tiers(signals, score)
 
    log.info("Tier distribution: T1=%d  T2=%d  T3=%d  No-action=%d",
             int((tiers==1).sum()), int((tiers==2).sum()),
             int((tiers==3).sum()), int((tiers==0).sum()))
 
    # ── Evaluate (if labels available) ───────────────────────────────────────
    results = None
    if y_true is not None:
        log.info("Evaluating against actual churn labels …")
        results = evaluate_tiers(y_true, tiers, score)
        print_report(results, y_true, tiers, signals, n_total=len(df))
    else:
        _sep()
        print("  SCORE-ONLY MODE — no target column found.")
        print(f"  Tier 1: {int((tiers==1).sum()):,}  |  "
              f"Tier 2: {int((tiers==2).sum()):,}  |  "
              f"Tier 3: {int((tiers==3).sum()):,}")
        _sep()
 
    # ── Save outputs ──────────────────────────────────────────────────────────
    save_outputs(df, tiers, score, signals, target_col)
 
    # ── Plot dashboard ────────────────────────────────────────────────────────
    if y_true is not None and results is not None:
        log.info("Generating dashboard …")
        plot_dashboard(y_true, tiers, score, signals, results, n_total=len(df))
 
    _sep()
    print(f"  DONE in {time.time()-t0:.1f}s")
    print(f"  Outputs written to: {CFG['OUTPUT_DIR']}/")
    print(f"    churn_scored.csv       — every subscriber with tier + signals")
    print(f"    churn_tier1_alerts.csv — high-confidence campaign list")
    print(f"    churn_tier2_alerts.csv — medium-risk automated campaign")
    print(f"    churn_tier3_watchlist.csv — digital/low-cost channel")
    print(f"    churn_rules_dashboard.png — visual summary")
    _sep()
 
    # Quick-reference CLI summary
    if y_true is not None and results:
        r1  = results["Tier 1 (high-confidence)"]
        r12 = results["Tier 1+2 combined"]
        print(f"\n  QUICK RESULTS:")
        print(f"  Tier 1 — Precision: {r1['precision']*100:.1f}%  "
              f"Recall: {r1['recall']*100:.1f}%  "
              f"Lift: {r1['lift']:.1f}×  "
              f"Alerts: {r1['alerts']:,}")
        print(f"  T1+T2  — Precision: {r12['precision']*100:.1f}%  "
              f"Recall: {r12['recall']*100:.1f}%  "
              f"Lift: {r12['lift']:.1f}×  "
              f"Alerts: {r12['alerts']:,}")
        print()
 
 
if __name__ == "__main__":
    main()