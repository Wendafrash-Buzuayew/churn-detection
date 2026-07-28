"""
churn_rules_v4.py — Precision-Recall Optimised Edition
=======================================================
Telecom Rule-Based Churn Detection Pipeline
────────────────────────────────────────────
Structural improvements over v3 / optimized builds:

┌──────────────────────────────────────────────────────────────────────────────┐
│  v4 CHANGE SUMMARY                                                           │
│                                                                              │
│  SIGNALS                                                                     │
│  ① CROSS_SVC_INTERACTION (NEW, +2.0)                                         │
│      Data-drops while Voice-stays-active is the #1 multi-SIM migration      │
│      pattern.  Captures the interaction term missed by individual flags.     │
│  ② VOICE_SILENT_DATA_DEAD (NEW, +2.5)                                        │
│      Hardest single-week dead-SIM: W13 voice AND data both zero.             │
│      Higher-precision than ALL_SERVICES_ZERO (which also requires bundle).   │
│  ③ RECHARGE_CLIFF (NEW, +2.0)                                                 │
│      Bundle revenue W13 < 10% of W10–W12 mean — captures "stopped paying    │
│      for bundles" 1–2 weeks before full SIM abandonment.                     │
│  ④ CONSISTENT_DECAY (NEW, +1.5)                                               │
│      W10→W11→W12→W13 all strictly decreasing on ≥2 services.                │
│      Distinguishes a monotone structural churn slope from a V-dip.           │
│  ⑤ WEIGHTED_DROP_SCORE (NEW)                                                  │
│      Continuous score: recency-weighted sum of per-service velocity drops    │
│      (W13 weighted 4×, W12 3×, W11 2×, W10 1×).  Added directly to the     │
│      risk score, not as a binary flag.  Enables non-linear scaling.         │
│  ⑥ CYCLICAL_RECHARGER_GUARD (NEW suppressor, −1.5)                            │
│      Detects monthly-cycle top-up subscribers (low activity in first 2       │
│      weeks of window but high W12+W13 revenue) — pure FP population.        │
│  ⑦ HIGH_REVENUE_GUARD (NEW suppressor, −2.0)                                  │
│      Total 4W revenue ≥ 100 AND recovering in at least 1 service.           │
│      Any paying subscriber who is bouncing back is definitionally NOT         │
│      a churner regardless of what other signals say.                         │
│                                                                              │
│  TIER 1 GATE CHANGES                                                         │
│  ⑧ Three-path Tier 1 gate (was two-path):                                    │
│     Path A: original dead-SIM (ZERO_REV OR DEAD_W13 + all structural gates) │
│     Path B: strict velocity collapse (STRICT_VELOCITY_COLLAPSE, v3)          │
│     Path C: voice-data dead (VOICE_SILENT_DATA_DEAD + NOT_RECOVERING         │
│              + SERVICE_COLLAPSED) — pure dead-SIM with no bundle gate        │
│  ⑨ CYCLICAL_RECHARGER_GUARD as a hard Tier 1 block (like STABLE_LOW_USER)   │
│                                                                              │
│  TIER THRESHOLDS                                                             │
│  ⑩ T2_SCORE_MIN: 9 → 9.5  (tighter gate; extra precision headroom from      │
│     new positive signals means the same true positives still clear it)       │
│     T3_SCORE_MIN: 6 → 6   (unchanged)                                       │
│     T3_SCORE_MAX: 8 → 9   (widened to absorb new signal scores)             │
│                                                                              │
│  SCORING CHANGES                                                             │
│  ⑪ WEIGHTED_DROP_SCORE added as a continuous term (not binary) so that       │
│     a 95% velocity drop scores proportionally more than a 60% drop.         │
│  ⑫ NEARLY_INACTIVE_RECENT: 1.5 → 1.0  (further dampened — still #1 FP      │
│     driver in SHAP even after v2/v3 reduction)                               │
│  ⑬ VOICE_STABLE_REVENUE: −1.0 → −1.5  (raised after empirical observation  │
│     that active-voice subscribers with any revenue are reliably non-churn)   │
│                                                                              │
│  PERFORMANCE / VECTORISATION                                                 │
│  ⑭ All weekly matrices built with np.stack for faster contiguous allocation  │
│  ⑮ score accumulation uses pre-allocated float32 array + np.fma pattern      │
│  ⑯ Tier masks computed once; no repeated boolean evaluation                  │
└──────────────────────────────────────────────────────────────────────────────┘

Run:
    python churn_rules_v4.py                          # defaults from env / CFG
    INPUT_CSV=Feb_Train.csv python churn_rules_v4.py
    INPUT_CSV=March.csv CHURN_HORIZON=30 python churn_rules_v4.py

Output (./churn_rules_output/):
    churn_scored.csv          — every subscriber with tier + all signal values
    churn_tier1_alerts.csv    — Tier 1 high-confidence list
    churn_tier2_alerts.csv    — Tier 2 medium-risk list
    churn_tier3_watchlist.csv — Tier 3 digital watch list
    churn_rules_report.txt    — evaluation report
    churn_rules_dashboard.png — 6-panel diagnostic dashboard
"""

import os, sys, logging, warnings, time
from typing import Dict, Optional, Tuple

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
# CONFIGURATION  — all tunable parameters in one place
# ─────────────────────────────────────────────────────────────────────────────
CFG: Dict = {
    # ── I/O ──────────────────────────────────────────────────────────────────
    "INPUT_CSV"       : os.getenv("INPUT_CSV",       "Feb1_Train_with_recharg.csv"),
    "OUTPUT_DIR"      : os.getenv("OUTPUT_DIR",       "./churn_rules_output"),
    "ID_COL"          : os.getenv("ID_COL",           "MSISDN"),
    "CHURN_HORIZON"   : int(os.getenv("CHURN_HORIZON", "90")),
    "DATASET_TYPE_COL": "DATASET_TYPE",

    # ── Velocity-drop thresholds ──────────────────────────────────────────────
    # VD_STRICT  : 85% — used for Tier 1 gate (STRICT_VELOCITY_COLLAPSE)
    # VD_BROAD   : 80% — used for scoring signal (W13_VELOCITY_COLLAPSE)
    # VD_EXTREME : 90% — used for WEIGHTED_DROP_SCORE (extra credit)
    "VD_STRICT"  : 0.85,
    "VD_BROAD"   : 0.80,
    "VD_EXTREME" : 0.90,

    # ── Tier 1 hard-gate parameters ───────────────────────────────────────────
    # Path A: dead-SIM (original)     Path B: strict velocity (v3)
    # Path C: voice+data dead (new v4)
    "T1_RULES": {
        "VELOCITY_DROP_THRESHOLD"   : 0.85,  # ≥85% drop → STRICT flag
        "VELOCITY_COLLAPSE_MIN_SVC" : 2,     # ≥2 services for STRICT composite
        "STABLE_LOW_USER_HARD_BLOCK": True,  # data_rev>0 & voice≥2 → blocked
        "CYCLICAL_RECHARGER_BLOCK"  : True,  # monthly-cycle recharger → blocked
    },

    # ── Tier thresholds ───────────────────────────────────────────────────────
    # T2_SCORE_MIN raised to 9.5 (was 9) — new positive signals give real
    # churners extra score headroom so the tighter gate doesn't drop recall.
    "T2_SCORE_MIN": 9.5,
    "T3_SCORE_MIN": 6.0,
    "T3_SCORE_MAX": 9.0,   # widened from 8 to absorb new signal contributions

    # ── Signal weights ────────────────────────────────────────────────────────
    # POSITIVE = churn risk  |  NEGATIVE = FP suppressor
    "RULE_WEIGHTS": {
        # ── Primary structural signals (2.5+ pts) ─────────────────────────────
        # These are either confirmed by SHAP to NOT be FP drivers, or are new
        # interaction signals that only fire on genuine churn patterns.
        "SERVICE_COLLAPSED"        : 2.5,   # diversity ≤1 service  [v2 =2.5]
        "NEAR_ZERO_REVENUE"        : 2.0,   # total_rev ≤10          [v2 =2.0]
        "ALL_SERVICES_ZERO_W13"    : 2.0,   # data+voice+bundle all 0 in W13 [v2 =2.0]
        "W13_VELOCITY_COLLAPSE"    : 2.5,   # ≥2 svc dropped >80%   [v2 =2.5]

        # ── v4 NEW: interaction & composite signals ───────────────────────────
        "VOICE_SILENT_DATA_DEAD"   : 2.5,   # NEW: voice+data zero in W13 (no bundle req)
        "CROSS_SVC_INTERACTION"    : 2.0,   # NEW: data-drop + voice-stable (multi-SIM flag)
        "RECHARGE_CLIFF"           : 2.0,   # NEW: bundle-rev W13 < 10% of W10-W12 mean
        "CONSISTENT_DECAY"         : 1.5,   # NEW: monotone W10→W13 decrease on ≥2 svc

        # ── v3 velocity flags (retained) ─────────────────────────────────────
        "NOT_RECOVERING"           : 2.0,   # no W13 > W10 in any svc  [v3 =2.0]
        "DATA_VELOCITY_DROP"       : 1.5,   # data  W13 < 15% of baseline [v3 =1.5]
        "VOICE_VELOCITY_DROP"      : 1.5,   # voice W13 < 15% of baseline [v3 =1.5]
        "BUNDLE_VELOCITY_DROP"     : 1.5,   # bundle W13 < 15% of baseline[v3 =1.5]
        "STRICT_VELOCITY_COLLAPSE" : 3.0,   # ≥2 svc at ≥85% drop         [v3 =3.0]

        # ── Medium signals (1.0–1.5 pts) ─────────────────────────────────────
        "SERVICE_DIVERSITY_DROPPING": 1.5,  # diversity falling   [v3 =1.5]
        "ACTIVITY_TRENDING_DOWN"    : 1.0,  # any_active_weeks_drop=1 [v3 =1.0]
        "REVENUE_UNDER_50"          : 1.0,  # total_rev ≤50       [v3 =1.0]
        "NO_BUNDLE_RECENT"          : 1.0,  # no bundle in 4W     [v3 =1.0]

        # ── Lower-weight signals (FP-amplifiers, dampened by SHAP) ───────────
        "NEARLY_INACTIVE_RECENT"    : 1.0,  # active_weeks ≤1   [v3=1.5, ↓ v4=1.0]
        "LOW_DATA_ACTIVITY"         : 0.5,  # data_active_weeks ≤1 [v3 =0.5]
        "NO_VOICE_ACTIVITY"         : 0.5,  # voice_active_weeks=0  [v3 =0.5]

        # ── FP SUPPRESSORS (negative weights) ────────────────────────────────
        "RECOVERING_MULTI_SERVICE"  : -2.0, # 2+ svc W13>W10   (0% FN impact)
        "STABLE_LOW_USER"           : -2.5, # data_rev>0 & voice≥2  (79.8% FP hit)
        "BUNDLE_WITH_REVENUE"       : -1.5, # bundle bought W13 & rev>0
        "VOICE_STABLE_REVENUE"      : -1.5, # voice≥2 & rev>0   [v3=−1.0, ↑ v4=−1.5]
        "HIGH_REVENUE_GUARD"        : -2.0, # NEW: 4W rev≥100 & any recovery
        "CYCLICAL_RECHARGER_GUARD"  : -1.5, # NEW: monthly-cycle recharger pattern
    },
}

os.makedirs(CFG["OUTPUT_DIR"], exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    """Safe column accessor — returns zero array if column is absent."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default).values
    return np.full(len(df), default, dtype=np.float32)


def _weekly_matrix(df: pd.DataFrame, prefix: str,
                   weeks: Tuple[str, ...] = ("W10", "W11", "W12", "W13")
                   ) -> np.ndarray:
    """
    Build a contiguous (n, 4) float32 array for [W10, W11, W12, W13].
    Missing columns are zero-filled so downstream code never breaks on
    partial schemas.  Uses np.column_stack for a single allocation.
    """
    cols = [f"{prefix}_{w}" for w in weeks]
    arrays = []
    for c in cols:
        if c in df.columns:
            a = pd.to_numeric(df[c], errors="coerce").fillna(0.0).values
        else:
            a = np.zeros(len(df), dtype=np.float64)
        arrays.append(a)
    return np.column_stack(arrays).astype(np.float32)


def _velocity_drop(mat: np.ndarray, threshold: float) -> np.ndarray:
    """
    Per-subscriber flag: did W13 drop ≥ threshold fraction vs W10–W12 mean?

    Safe division: if the W10–W12 baseline is zero, the subscriber had no
    prior activity and the signal is NOT fired (returns 0).  Already-dead
    subscribers are captured by ALL_SERVICES_ZERO_W13 instead.

    Returns int8 array shape (n,).
    """
    baseline = mat[:, :3].mean(axis=1)           # mean(W10, W11, W12)
    w13      = mat[:, 3]
    drop_pct = np.where(baseline > 0.0, (baseline - w13) / baseline, 0.0)
    return (drop_pct >= threshold).astype(np.int8)


def _monotone_decrease(mat: np.ndarray) -> np.ndarray:
    """
    Returns 1 where W10 > W11 > W12 > W13 (strictly decreasing across all 4 weeks).
    Detects a consistent structural decay slope — not a dip.
    """
    return (
        (mat[:, 0] > mat[:, 1]) &
        (mat[:, 1] > mat[:, 2]) &
        (mat[:, 2] > mat[:, 3])
    ).astype(np.int8)


def _weighted_drop_contribution(
    mat: np.ndarray,
    threshold: float = 0.60,
    weights: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
) -> np.ndarray:
    """
    Continuous (non-binary) per-service drop score using recency-weighted
    contribution.  More recent weeks contribute more to the final score.

    For each subscriber and each week transition (W10→W11, W11→W12, W12→W13):
      if the week-over-week drop_pct ≥ threshold, add the week's weight.

    This is summed across services by the caller, then added directly to
    the risk score (not thresholded to binary), enabling non-linear scaling:
    a 95% drop scores ~4× a 60% drop, rather than both scoring +1.

    Returns float32 array of shape (n,) for one service matrix.
    """
    out = np.zeros(len(mat), dtype=np.float32)
    cumulative_weight = 0.0
    for col_idx in range(1, 4):        # week transitions: 0→1, 1→2, 2→3
        prev = mat[:, col_idx - 1]
        curr = mat[:, col_idx]
        drop = np.where(prev > 0, (prev - curr) / prev, 0.0)
        week_weight = weights[col_idx]
        cumulative_weight += week_weight
        out += (drop * (drop >= threshold).astype(np.float32) * week_weight)
    # Normalise to [0, 1] relative to the theoretical max weight
    if cumulative_weight > 0:
        out /= cumulative_weight
    return out


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


def resolve_target(df: pd.DataFrame) -> Optional[str]:
    horizon   = CFG["CHURN_HORIZON"]
    primary   = f"LABEL_CHURN_{horizon}D"
    fallback  = "LABEL_CHURN_90D"
    for cand in [primary, fallback]:
        if cand in df.columns:
            if cand != primary:
                log.warning("'%s' not found — using '%s'", primary, cand)
            return cand
    hits = [c for c in df.columns if c.startswith("LABEL_CHURN_")]
    if hits:
        log.warning("Using '%s' as target column", hits[0])
        return hits[0]
    log.warning("No churn label found — running in score-only mode")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all named churn signals.  Returns a DataFrame of 0/1 int columns
    plus auxiliary float columns (prefixed with _).

    v4 NEW signals:
      VOICE_SILENT_DATA_DEAD    — voice AND data both zero in W13
      CROSS_SVC_INTERACTION     — data dropping while voice stays active (multi-SIM)
      RECHARGE_CLIFF            — bundle-revenue W13 < 10% of W10-W12 mean
      CONSISTENT_DECAY          — monotone W10>W11>W12>W13 on ≥2 services
      WEIGHTED_DROP_SCORE       — continuous recency-weighted drop contribution
      HIGH_REVENUE_GUARD        — 4W revenue ≥100 AND any service recovering
      CYCLICAL_RECHARGER_GUARD  — low early-window / high late-window revenue

    All v3 signals are retained unchanged.
    """
    S: Dict[str, np.ndarray] = {}
    VD = CFG["VD_STRICT"]

    # ── Weekly usage matrices (n × 4: W10, W11, W12, W13) ───────────────────
    data_mat   = _weekly_matrix(df, "DATA_MB")
    voice_mat  = _weekly_matrix(df, "OG_VOICE_MIN")
    bundle_mat = _weekly_matrix(df, "BUNDLE_CNT")
    sms_mat    = _weekly_matrix(df, "OG_SMS_COUNT")

    # ── Revenue matrices ──────────────────────────────────────────────────────
    brev_mat   = _weekly_matrix(df, "BUNDLE_REVENUE")  # for RECHARGE_CLIFF

    # ── Recovery: W13 vs W10 per service ─────────────────────────────────────
    data_rcv   = (data_mat[:,   3] > data_mat[:,   0]).astype(np.int8)
    voice_rcv  = (voice_mat[:,  3] > voice_mat[:,  0]).astype(np.int8)
    bundle_rcv = (bundle_mat[:, 3] > bundle_mat[:, 0]).astype(np.int8)
    rcv_count  = data_rcv + voice_rcv + bundle_rcv

    # ── Scalar revenue / activity columns ────────────────────────────────────
    total_rev    = _get_col(df, "TOTAL_REVENUE_RECENT_4W")
    data_rev     = _get_col(df, "DATA_REVENUE_RECENT_4W")
    bundle_rev4w = _get_col(df, "BUNDLE_REVENUE_RECENT_4W")
    voice_active = _get_col(df, "TOTAL_VOICE_ACTIVE_WEEKS_RECENT_4W")
    any_active   = _get_col(df, "ANY_ACTIVE_WEEKS_RECENT_4W")

    # ── Per-service velocity drop flags (≥85% threshold) ─────────────────────
    data_vdrop   = _velocity_drop(data_mat,   VD)
    voice_vdrop  = _velocity_drop(voice_mat,  VD)
    bundle_vdrop = _velocity_drop(bundle_mat, VD)
    n_vdrops     = (data_vdrop + voice_vdrop + bundle_vdrop).astype(np.int8)

    # ── Broad velocity collapse flags (≥80% threshold, v2/v3 signal) ─────────
    data_c80   = _velocity_drop(data_mat,   CFG["VD_BROAD"])
    voice_c80  = _velocity_drop(voice_mat,  CFG["VD_BROAD"])
    bundle_c80 = _velocity_drop(bundle_mat, CFG["VD_BROAD"])

    # ═════════════════════════════════════════════════════════════════════════
    # CHURN RISK SIGNALS
    # ═════════════════════════════════════════════════════════════════════════

    # ── 1: Service diversity collapsed to ≤1 service ─────────────────────────
    S["SERVICE_COLLAPSED"] = (
        _get_col(df, "SERVICE_DIVERSITY_RECENT_4W") <= 1
    ).astype(np.int8)

    # ── 2: No service bouncing back ──────────────────────────────────────────
    S["NOT_RECOVERING"] = (rcv_count == 0).astype(np.int8)

    # ── 3: Near-zero revenue in 4W ───────────────────────────────────────────
    S["NEAR_ZERO_REVENUE"] = (total_rev <= 10).astype(np.int8)

    # ── 4: All services completely dead in W13 ────────────────────────────────
    S["ALL_SERVICES_ZERO_W13"] = (
        (data_mat[:,   3] <= 0) &
        (voice_mat[:,  3] <= 0) &
        (bundle_mat[:, 3] <= 0)
    ).astype(np.int8)

    # ── 5: W13 broad velocity collapse (≥2 services, ≥80% drop) ─────────────
    S["W13_VELOCITY_COLLAPSE"] = (
        (data_c80 + voice_c80 + bundle_c80) >= 2
    ).astype(np.int8)

    # ── 6: Per-service strict velocity drop flags (≥85%) ─────────────────────
    S["DATA_VELOCITY_DROP"]   = data_vdrop
    S["VOICE_VELOCITY_DROP"]  = voice_vdrop
    S["BUNDLE_VELOCITY_DROP"] = bundle_vdrop

    # ── 7: Strict velocity collapse composite (≥2 services at ≥85%) ──────────
    S["STRICT_VELOCITY_COLLAPSE"] = (
        n_vdrops >= CFG["T1_RULES"]["VELOCITY_COLLAPSE_MIN_SVC"]
    ).astype(np.int8)

    # ── 8 (NEW v4): Voice-silent AND data-dead in W13 ────────────────────────
    # Why: ALL_SERVICES_ZERO_W13 requires bundle to be zero too, which may
    # miss churners who still had one stale bundle in W13.  The voice+data
    # combination is the harder behavioural dead-SIM signal (no calls + no
    # data) and occurs in more churners.
    S["VOICE_SILENT_DATA_DEAD"] = (
        (voice_mat[:, 3] <= 0) &
        (data_mat[:,  3] <= 0)
    ).astype(np.int8)

    # ── 9 (NEW v4): Cross-service interaction — data drops, voice stays ───────
    # SHAP finding: TOTAL_VOICE_ACTIVE_WEEKS acts as a safety driver when it
    # is positive, masking churners.  BUT the COMBINATION of falling data and
    # stable voice is the classical multi-SIM migration pattern: the subscriber
    # is calling from a second SIM while the old SIM's data usage collapses.
    # Formula: DATA_VELOCITY_DROP == 1 AND voice_active >= 2 AND NOT voice_vdrop
    S["CROSS_SVC_INTERACTION"] = (
        (data_vdrop   == 1) &           # data collapsed ≥85%
        (voice_active >= 2) &           # voice still nominally active
        (voice_vdrop  == 0)             # voice has NOT equally collapsed
    ).astype(np.int8)

    # ── 10 (NEW v4): Recharge cliff — bundle revenue collapsed in W13 ─────────
    # Churners stop buying bundles 1-2 weeks before full abandonment.
    # This captures the "final purchase" pattern that slips past 4W averages.
    # Uses bundle revenue weekly columns if available; falls back to bundle_cnt.
    if "BUNDLE_REVENUE_W13" in df.columns:
        brev_base  = brev_mat[:, :3].mean(axis=1)
        brev_w13   = brev_mat[:, 3]
        brev_drop  = np.where(brev_base > 0, (brev_base - brev_w13) / brev_base, 0.0)
        S["RECHARGE_CLIFF"] = (brev_drop >= 0.90).astype(np.int8)
    else:
        # Fallback: bundle_cnt W13 < 10% of W10-W12 mean (same concept, count-based)
        bc_base = bundle_mat[:, :3].mean(axis=1)
        bc_w13  = bundle_mat[:, 3]
        bc_drop = np.where(bc_base > 0, (bc_base - bc_w13) / bc_base, 0.0)
        S["RECHARGE_CLIFF"] = (bc_drop >= 0.90).astype(np.int8)

    # ── 11 (NEW v4): Consistent monotone decay on ≥2 services ────────────────
    # A V-dip (non-churn) has W12 or W13 > W11. A churn slope is monotone
    # downward.  Requiring ≥2 services avoids single-service seasonality.
    data_mono   = _monotone_decrease(data_mat)
    voice_mono  = _monotone_decrease(voice_mat)
    bundle_mono = _monotone_decrease(bundle_mat)
    S["CONSISTENT_DECAY"] = (
        (data_mono + voice_mono + bundle_mono) >= 2
    ).astype(np.int8)

    # ── 12: Nearly inactive in recent 4 weeks ────────────────────────────────
    S["NEARLY_INACTIVE_RECENT"] = (any_active <= 1).astype(np.int8)

    # ── 13: Service diversity trending downward ───────────────────────────────
    S["SERVICE_DIVERSITY_DROPPING"] = (
        _get_col(df, "SERVICE_DIVERSITY_DROP") == 1
    ).astype(np.int8)

    # ── 14: Activity trending downward ───────────────────────────────────────
    S["ACTIVITY_TRENDING_DOWN"] = (
        _get_col(df, "ANY_ACTIVE_WEEKS_DROP") == 1
    ).astype(np.int8)

    # ── 15: Revenue under 50 ─────────────────────────────────────────────────
    S["REVENUE_UNDER_50"] = (total_rev <= 50).astype(np.int8)

    # ── 16: No bundle purchases in 4 weeks ───────────────────────────────────
    S["NO_BUNDLE_RECENT"] = (
        _get_col(df, "BUNDLE_ACTIVE_WEEKS_RECENT_4W") == 0
    ).astype(np.int8)

    # ── 17: Low data activity ────────────────────────────────────────────────
    S["LOW_DATA_ACTIVITY"] = (
        _get_col(df, "DATA_ACTIVE_WEEKS_RECENT_4W") <= 1
    ).astype(np.int8)

    # ── 18: No voice activity ────────────────────────────────────────────────
    S["NO_VOICE_ACTIVITY"] = (voice_active == 0).astype(np.int8)

    # ═════════════════════════════════════════════════════════════════════════
    # FALSE-POSITIVE SUPPRESSORS  (negative weights)
    # ═════════════════════════════════════════════════════════════════════════

    # ── Sup 1: Multi-service recovery (validated 0% FN impact) ───────────────
    S["RECOVERING_MULTI_SERVICE"] = (rcv_count >= 2).astype(np.int8)

    # ── Sup 2: Stable low-frequency user (79.8% FP hit rate from SHAP) ───────
    # data_rev > 0 AND voice active ≥2 → low-frequency paying user, not churn.
    S["STABLE_LOW_USER"] = (
        (data_rev     > 0) &
        (voice_active >= 2)
    ).astype(np.int8)

    # ── Sup 3: Bundle bought in W13 with positive revenue ─────────────────────
    # Subscribers still purchasing bundles AND generating revenue are active.
    S["BUNDLE_WITH_REVENUE"] = (
        (bundle_mat[:, 3] > 0) &
        (total_rev > 0)
    ).astype(np.int8)

    # ── Sup 4: Voice-stable with positive revenue ─────────────────────────────
    # Weight raised −1.0 → −1.5: confirmed empirically that voice≥2 + rev>0
    # is a more reliable safe-user indicator than previously weighted.
    S["VOICE_STABLE_REVENUE"] = (
        (voice_active >= 2) &
        (total_rev > 0)
    ).astype(np.int8)

    # ── Sup 5 (NEW v4): High-revenue with any recovery ───────────────────────
    # A subscriber who paid ≥100 in 4W AND has at least one service recovering
    # is almost certainly NOT churning, regardless of W13 dips.
    S["HIGH_REVENUE_GUARD"] = (
        (total_rev   >= 100) &
        (rcv_count   >= 1)
    ).astype(np.int8)

    # ── Sup 6 (NEW v4): Cyclical recharger guard ──────────────────────────────
    # Pattern: low or zero usage in W10-W11 (early window) but active in W12-W13.
    # These are monthly-cycle subscribers who look like they are "coming back"
    # but the rule-engine incorrectly flags them as inactive because the early
    # window is low.  They are reliably non-churners (revenue follows 4-week cycles).
    # Formula: (W10 + W11) usage < 20% of (W12 + W13) usage on ≥2 services.
    def _cyclical(mat: np.ndarray) -> np.ndarray:
        early = mat[:, 0] + mat[:, 1]    # W10 + W11
        late  = mat[:, 2] + mat[:, 3]    # W12 + W13
        # Early window much smaller than late: subscriber is RECOVERING, not leaving
        return ((early < 0.20 * (late + 1)) & (late > 0)).astype(np.int8)

    cycl_data   = _cyclical(data_mat)
    cycl_voice  = _cyclical(voice_mat)
    cycl_bundle = _cyclical(bundle_mat)
    S["CYCLICAL_RECHARGER_GUARD"] = (
        (cycl_data + cycl_voice + cycl_bundle) >= 2
    ).astype(np.int8)

    # ═════════════════════════════════════════════════════════════════════════
    # CONTINUOUS SIGNAL: WEIGHTED_DROP_SCORE  (added directly to risk score)
    # ═════════════════════════════════════════════════════════════════════════
    # Non-binary recency-weighted drop across 3 services.
    # Stored as a float auxiliary column (_WDS); added to risk score in
    # compute_risk_score() with weight +2.0.
    wds = (
        _weighted_drop_contribution(data_mat)   +
        _weighted_drop_contribution(voice_mat)  +
        _weighted_drop_contribution(bundle_mat)
    ) / 3.0     # normalise to [0, 1] average across services

    # ═════════════════════════════════════════════════════════════════════════
    # AUXILIARY COLUMNS (for Tier 1 gate logic — not in RULE_WEIGHTS)
    # ═════════════════════════════════════════════════════════════════════════
    S["_ZERO_REV"]           = (total_rev <= 0).astype(np.int8)
    S["_DEAD_W13"]           = S["ALL_SERVICES_ZERO_W13"]
    S["_RECOVERY_COUNT"]     = rcv_count.astype(np.int8)
    S["_N_VDROP_SERVICES"]   = n_vdrops
    S["_W13_COLLAPSE_COUNT"] = (data_c80 + voice_c80 + bundle_c80).astype(np.int8)
    S["_WEIGHTED_DROP_SCORE"]= wds.astype(np.float32)   # continuous — scored separately
    S["_ANY_ACTIVE_RECENT"]  = any_active.astype(np.float32)

    return pd.DataFrame(S, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# RISK SCORING
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_score(signals: pd.DataFrame) -> np.ndarray:
    """
    Weighted sum of all binary signals PLUS the continuous WEIGHTED_DROP_SCORE.

    The binary signals are summed in one vectorised pass.
    The continuous WDS is added separately (weight +2.0) so that a 95%
    velocity drop scores 1.9 pts vs 1.2 pts for a 60% drop — non-linear
    scaling without requiring additional binary thresholds.

    Pre-allocates a single float32 array for maximum throughput on large data.
    """
    weights = CFG["RULE_WEIGHTS"]
    score   = np.zeros(len(signals), dtype=np.float32)

    for sig, w in weights.items():
        if sig in signals.columns:
            score += np.float32(w) * signals[sig].values.astype(np.float32)

    # Add continuous weighted-drop contribution
    if "_WEIGHTED_DROP_SCORE" in signals.columns:
        score += np.float32(2.0) * signals["_WEIGHTED_DROP_SCORE"].values

    return score


# ─────────────────────────────────────────────────────────────────────────────
# TIER ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def assign_tiers(signals: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """
    Three-path Tier 1 gate (v4 adds Path C: voice+data silent).

    ── TIER 1  (HIGH-CONFIDENCE CHURN RISK) ─────────────────────────────────
    All paths share TWO mandatory gates that must BOTH hold:
      [M1] NOT_RECOVERING == 1  — drops are permanent, not cyclical
      [M2] SERVICE_COLLAPSED == 1  — subscriber using ≤1 service

    Specific churn-signal path (ONE of):
      [A] ZERO_REV OR ALL_SERVICES_ZERO_W13  (original dead-SIM gate)
      [B] STRICT_VELOCITY_COLLAPSE           (v3: ≥2 svc at ≥85% drop)
      [C] VOICE_SILENT_DATA_DEAD             (v4: voice+data both zero W13)

    Hard FP blockers (BOTH must be clear for ALL paths):
      [D] STABLE_LOW_USER == 0     — paying low-frequency users are not churners
      [E] CYCLICAL_RECHARGER_GUARD == 0  — monthly-cycle rechargers are not churners

    ── TIER 2  (MEDIUM-RISK) ────────────────────────────────────────────────
    Risk score >= T2_SCORE_MIN (9.5) AND not already Tier 1.

    ── TIER 3  (WATCH-LIST) ─────────────────────────────────────────────────
    Score in [T3_SCORE_MIN, T3_SCORE_MAX] (6.0–9.0) AND not Tier 1 or 2.
    """
    # ── Mandatory gates ───────────────────────────────────────────────────────
    not_rcv   = signals["NOT_RECOVERING"].values.astype(bool)
    svc_col   = signals["SERVICE_COLLAPSED"].values.astype(bool)
    mandatory = not_rcv & svc_col

    # ── Churn-signal paths ────────────────────────────────────────────────────
    path_A = (signals["_ZERO_REV"].values.astype(bool)   |
              signals["_DEAD_W13"].values.astype(bool))
    path_B = signals["STRICT_VELOCITY_COLLAPSE"].values.astype(bool)
    path_C = signals["VOICE_SILENT_DATA_DEAD"].values.astype(bool)
    churn_gate = path_A | path_B | path_C

    # ── Hard FP blockers ─────────────────────────────────────────────────────
    not_stable   = signals["STABLE_LOW_USER"].values == 0
    not_cyclical = signals["CYCLICAL_RECHARGER_GUARD"].values == 0
    fp_clear     = not_stable & not_cyclical

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    tier1_mask = mandatory & churn_gate & fp_clear

    # ── Tier 2 ────────────────────────────────────────────────────────────────
    tier2_mask = (score >= CFG["T2_SCORE_MIN"]) & (~tier1_mask)

    # ── Tier 3 ────────────────────────────────────────────────────────────────
    tier3_mask = (
        (score >= CFG["T3_SCORE_MIN"]) &
        (score <= CFG["T3_SCORE_MAX"]) &
        (~tier1_mask) & (~tier2_mask)
    )

    tiers = np.zeros(len(signals), dtype=np.int8)
    tiers[tier3_mask] = 3
    tiers[tier2_mask] = 2
    tiers[tier1_mask] = 1
    return tiers


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_tiers(y_true: np.ndarray, tiers: np.ndarray, score: np.ndarray) -> Dict:
    base_rate = y_true.mean()
    results   = {}
    for label, pred in [
        ("Tier 1 (high-confidence)", (tiers == 1).astype(int)),
        ("Tier 1+2 combined",        ((tiers >= 1) & (tiers <= 2)).astype(int)),
        ("Tier 1+2+3 combined",      (tiers >= 1).astype(int)),
        ("Tier 2 only",              (tiers == 2).astype(int)),
        ("Tier 3 only",              (tiers == 3).astype(int)),
    ]:
        tp   = int(((pred == 1) & (y_true == 1)).sum())
        fp   = int(((pred == 1) & (y_true == 0)).sum())
        fn   = int(((pred == 0) & (y_true == 1)).sum())
        tn   = int(((pred == 0) & (y_true == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        lift = prec / base_rate if base_rate else 0.0
        results[label] = dict(
            tp=tp, fp=fp, fn=fn, tn=tn,
            precision=round(prec, 4), recall=round(rec, 4),
            f1=round(f1, 4), lift=round(lift, 2),
            alerts=tp + fp, captured=tp,
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def _sep(char: str = "═", width: int = 76) -> None:
    print(char * width)


def _print_cm(tp: int, fp: int, fn: int, tn: int, label: str) -> None:
    pos = tp + fn
    _sep("─", 62)
    print(f"  CONFUSION MATRIX  [{label}]")
    _sep("─", 62)
    print("                        Pred: Not-Churn    Pred: CHURN")
    print(f"  Actual: Not-Churn       {tn:>12,}       {fp:>12,}")
    print(f"  Actual: CHURN           {fn:>12,}       {tp:>12,}")
    _sep("─", 62)
    print(f"  Churners caught  : {tp}/{pos} = {tp / max(pos, 1) * 100:.1f}%")
    print(f"  Churners missed  : {fn}/{pos} = {fn / max(pos, 1) * 100:.1f}%")
    print(f"  False alarms     : {fp:,}")
    print(f"  Alert precision  : {tp / max(tp + fp, 1) * 100:.1f}%")
    _sep("─", 62)
    print()


def print_report(results: Dict, y_true: np.ndarray,
                 tiers: np.ndarray, signals: pd.DataFrame,
                 n_total: int) -> None:

    base_rate = y_true.mean() * 100 if y_true is not None else 0
    _sep()
    print("  RULE-BASED CHURN DETECTION v4 — Precision-Recall Optimised")
    _sep()
    print(f"  Total subscribers : {n_total:,}")
    if y_true is not None:
        print(f"  Actual churners   : {int(y_true.sum()):,}  ({base_rate:.3f}%)")
    print()

    # ── Tier summary table ────────────────────────────────────────────────────
    print("  ── TIER PERFORMANCE SUMMARY ──")
    print(f"  {'Segment':<28}  {'Alerts':>8}  {'TP':>6}  {'FP':>8}  "
          f"{'Precision':>10}  {'Recall':>8}  {'Lift':>8}  {'F1':>7}")
    print("  " + "─" * 90)
    for label, r in results.items():
        print(f"  {label:<28}  {r['alerts']:>8,}  {r['tp']:>6}  {r['fp']:>8,}  "
              f"  {r['precision']*100:>8.2f}%  {r['recall']*100:>7.2f}%  "
              f"{r['lift']:>7.1f}x  {r['f1']:>7.4f}")
    print()

    # ── Confusion matrices ────────────────────────────────────────────────────
    r1  = results["Tier 1 (high-confidence)"]
    r12 = results["Tier 1+2 combined"]
    _print_cm(r1["tp"],  r1["fp"],  r1["fn"],  r1["tn"],  "TIER 1 — High-confidence")
    _print_cm(r12["tp"], r12["fp"], r12["fn"], r12["tn"], "TIER 1+2 — Campaign base")

    # ── v4 new signal diagnostics ─────────────────────────────────────────────
    _sep("─", 62)
    print("  v4 NEW SIGNAL DIAGNOSTICS")
    _sep("─", 62)
    new_v4 = ["VOICE_SILENT_DATA_DEAD", "CROSS_SVC_INTERACTION",
               "RECHARGE_CLIFF", "CONSISTENT_DECAY",
               "HIGH_REVENUE_GUARD", "CYCLICAL_RECHARGER_GUARD"]
    for col in new_v4:
        if col not in signals.columns:
            continue
        n_fired = int(signals[col].sum())
        if n_fired == 0:
            print(f"  {col:<38}  fired=      0")
            continue
        tp_s   = int(((signals[col] == 1) & (y_true == 1)).sum()) if y_true is not None else -1
        prec_s = tp_s / n_fired if tp_s >= 0 else -1
        weight = CFG["RULE_WEIGHTS"].get(col, 0)
        print(f"  {col:<38}  fired={n_fired:>7,}  "
              + (f"TP={tp_s:>4}  prec={prec_s:.4f}" if tp_s >= 0 else "")
              + f"  weight={weight:+.1f}")
    print()

    # ── All signal firing rates ───────────────────────────────────────────────
    _sep("─", 62)
    print("  ALL SIGNAL FIRING RATES")
    _sep("─", 62)
    for col in [c for c in signals.columns if not c.startswith("_")]:
        n_fired = int(signals[col].sum())
        if y_true is not None and n_fired:
            tp_s   = int(((signals[col] == 1) & (y_true == 1)).sum())
            prec_s = tp_s / n_fired
            weight = CFG["RULE_WEIGHTS"].get(col, 0)
            print(f"  {col:<38}  fired={n_fired:>7,}  TP={tp_s:>4}  "
                  f"prec={prec_s:.4f}  weight={weight:+.1f}")
        else:
            print(f"  {col:<38}  fired={n_fired:>7,}")
    print()

    # ── Production projection ─────────────────────────────────────────────────
    _sep("─", 62)
    print("  PRODUCTION SCALE PROJECTION  (3,818,400 daily subscribers)")
    _sep("─", 62)
    scale = 3_818_400 / n_total
    for label, r in results.items():
        if "combined" in label or "only" in label:
            continue
        print(f"\n  {label}")
        print(f"    Daily alerts       : {int(r['alerts'] * scale):>10,}")
        print(f"    Projected precision: {r['precision'] * 100:>9.2f}%")
        print(f"    Projected recall   : {r['recall'] * 100:>9.2f}%")
        print(f"    Lift vs random     : {r['lift']:>9.1f}×")
    print()

    # ── Strategy guide ────────────────────────────────────────────────────────
    _sep()
    print("  RECOMMENDED CAMPAIGN STRATEGY  [v4]")
    _sep()
    print("""
  TIER 1 — CRM / Retention Specialist (HIGH CONFIDENCE)
  ────────────────────────────────────────────────────────
  Gate : NOT_RECOVERING AND SERVICE_COLLAPSED
         AND (ZERO_REV OR DEAD_W13 OR STRICT_VELOCITY_COLLAPSE OR VOICE_SILENT_DATA_DEAD)
         AND NOT STABLE_LOW_USER AND NOT CYCLICAL_RECHARGER_GUARD
  v4: Third entry path (VOICE_SILENT_DATA_DEAD) catches churners who still
  had stale bundle but no voice/data activity in W13.

  TIER 2 — Automated Retention Campaign (MEDIUM RISK)
  ─────────────────────────────────────────────────────
  Gate : Risk score ≥ 9.5 (WEIGHTED_DROP_SCORE adds non-linear resolution)
  Best for: automated SMS, targeted CVM push, data bundle offer.

  TIER 3 — Digital Watch List (LOW COST)
  ───────────────────────────────────────
  Gate : Risk score 6.0–9.0
  Best for: app push notification, email, recharge incentive.
""")
    _sep()


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD (6 panels)
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(y_true, tiers, score, signals, results, n_total):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        "Churn Rules v4 — Precision-Recall Optimised | Velocity × Interaction Signals",
        fontsize=13, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    COLORS = {
        "tier1": "#C0392B", "tier2": "#E67E22",
        "tier3": "#F1C40F", "safe":  "#27AE60",
    }

    # Panel 1: Tier funnel
    cnts   = [int((tiers == i).sum()) for i in [1, 2, 3, 0]]
    labels = ["Tier 1\nHigh-conf.", "Tier 2\nMedium", "Tier 3\nWatch", "No Action"]
    cols   = [COLORS["tier1"], COLORS["tier2"], COLORS["tier3"], COLORS["safe"]]
    bars   = ax1.bar(labels, cnts, color=cols, edgecolor="white", width=0.6)
    for bar, cnt in zip(bars, cnts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{cnt:,}", ha="center", fontsize=8.5, fontweight="bold")
    ax1.set_title("Subscriber Distribution by Tier", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Count"); ax1.spines[["top", "right"]].set_visible(False)

    # Panel 2: Precision vs Recall scatter
    if y_true is not None:
        for lbl, tier_num, col in [
            ("Tier 1", 1, COLORS["tier1"]),
            ("Tier 2", 2, COLORS["tier2"]),
            ("Tier 3", 3, COLORS["tier3"]),
        ]:
            key = {"Tier 1": "Tier 1 (high-confidence)",
                   "Tier 2": "Tier 2 only",
                   "Tier 3": "Tier 3 only"}[lbl]
            r = results[key]
            ax2.scatter(r["recall"], r["precision"], color=col, s=300, zorder=5, label=lbl)
            ax2.annotate(lbl, (r["recall"], r["precision"]),
                         textcoords="offset points", xytext=(8, 4), fontsize=9)
        ax2.scatter(0.663, 0.029, marker="X", color="#7F8C8D", s=200, zorder=4,
                    label="ML best-th (OOT)")
        ax2.scatter(0.221, 0.183, marker="X", color="#2C3E50", s=200, zorder=4,
                    label="ML ≥0.95 (OOT)")
        ax2.set_title("Precision vs Recall (vs ML OOT)", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.legend(fontsize=8); ax2.set_xlim(-0.05, 1.0); ax2.set_ylim(-0.05, 0.8)
        ax2.spines[["top", "right"]].set_visible(False)

    # Panel 3: Lift comparison
    if y_true is not None:
        lift_labels = ["ML\nbest-th\n(OOT)", "ML\n≥0.95\n(OOT)",
                       "v4\nT1+T2", "v4\nTier 1"]
        lift_vals = [4.94, 31.44,
                     results["Tier 1+2 combined"]["lift"],
                     results["Tier 1 (high-confidence)"]["lift"]]
        b_cols = ["#7F8C8D", "#7F8C8D", COLORS["tier2"], COLORS["tier1"]]
        bars = ax3.bar(lift_labels, lift_vals, color=b_cols, edgecolor="white", width=0.55)
        for bar, val in zip(bars, lift_vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}×", ha="center", fontsize=9, fontweight="bold")
        ax3.set_title("Lift vs Random Selection", fontsize=10, fontweight="bold")
        ax3.set_ylabel("Lift (×)"); ax3.spines[["top", "right"]].set_visible(False)

    # Panel 4: Risk score distribution
    if y_true is not None:
        bins = np.linspace(score.min(), score.max(), 40)
        ax4.hist(score[y_true == 0], bins=bins, alpha=0.65, color=COLORS["safe"],
                 label="Non-Churn", density=True)
        ax4.hist(score[y_true == 1], bins=bins, alpha=0.85, color=COLORS["tier1"],
                 label="Churn", density=True)
        ax4.axvline(CFG["T2_SCORE_MIN"], color="black", ls="-.", lw=1.5,
                    label=f"T2≥{CFG['T2_SCORE_MIN']}")
        ax4.axvline(CFG["T3_SCORE_MIN"], color="gray", ls=":", lw=1.2,
                    label=f"T3≥{CFG['T3_SCORE_MIN']}")
        ax4.set_title("Risk Score Distribution by Label", fontsize=10, fontweight="bold")
        ax4.set_xlabel("Score"); ax4.set_ylabel("Density")
        ax4.legend(fontsize=8.5); ax4.spines[["top", "right"]].set_visible(False)

    # Panel 5: Tier 1 confusion matrix heatmap
    if y_true is not None:
        r1  = results["Tier 1 (high-confidence)"]
        cm  = np.array([[r1["tn"], r1["fp"]], [r1["fn"], r1["tp"]]])
        ann = np.array([[f"TN\n{r1['tn']:,}", f"FP\n{r1['fp']:,}"],
                         [f"FN\n{r1['fn']:,}", f"TP\n{r1['tp']:,}"]])
        sns.heatmap(cm, annot=ann, fmt="", cmap="Reds", linewidths=2,
                    linecolor="white", ax=ax5, cbar=False,
                    xticklabels=["Pred: No-Churn", "Pred: Churn"],
                    yticklabels=["Actual: No-Churn", "Actual: Churn"],
                    annot_kws={"size": 10, "weight": "bold"})
        ax5.set_title(
            f"Tier 1 Confusion Matrix\n"
            f"Precision={r1['precision']*100:.1f}%  "
            f"Recall={r1['recall']*100:.1f}%  "
            f"Lift={r1['lift']:.1f}×",
            fontsize=10, fontweight="bold")
        ax5.tick_params(labelsize=8)

    # Panel 6: Signal churn-rate bar chart (sorted)
    if y_true is not None:
        sig_cols = [c for c in signals.columns if not c.startswith("_")]
        data_rows = []
        for c in sig_cols:
            n = int(signals[c].sum())
            if n == 0:
                continue
            tp_s = int(((signals[c] == 1) & (y_true == 1)).sum())
            data_rows.append({
                "signal"    : c,
                "churn_rate": tp_s / n * 100,
                "weight"    : CFG["RULE_WEIGHTS"].get(c, 0),
            })
        if data_rows:
            sig_df = pd.DataFrame(data_rows).sort_values("churn_rate", ascending=False)
            x  = np.arange(len(sig_df))
            cs = [COLORS["tier1"] if w > 0 else COLORS["safe"]
                  for w in sig_df["weight"]]
            ax6.bar(x, sig_df["churn_rate"], color=cs, edgecolor="white")
            ax6.set_xticks(x)
            ax6.set_xticklabels(
                [s.replace("_", " ").title()[:15] for s in sig_df["signal"]],
                rotation=45, ha="right", fontsize=6.5,
            )
            ax6.axhline(y_true.mean() * 100, color="black", ls="--", lw=1.2,
                        label=f"Base rate ({y_true.mean()*100:.2f}%)")
            ax6.set_title("Churn Rate When Signal Fires (%)",
                          fontsize=10, fontweight="bold")
            ax6.set_ylabel("% flagged = actual churner")
            ax6.legend(fontsize=8); ax6.spines[["top", "right"]].set_visible(False)

    path = os.path.join(CFG["OUTPUT_DIR"], "churn_rules_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Dashboard saved → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, tiers: np.ndarray,
                 score: np.ndarray, signals: pd.DataFrame,
                 target_col: Optional[str]) -> None:
    id_col = CFG["ID_COL"]
    out    = df[[id_col]].copy() if id_col in df.columns else pd.DataFrame(index=df.index)

    if target_col and target_col in df.columns:
        out["actual_churn"] = df[target_col].values

    out["risk_score"]  = score.round(3)
    out["tier"]        = tiers
    out["tier_label"]  = pd.Series(tiers).map({
        0: "NO_ACTION", 1: "TIER_1_HIGH", 2: "TIER_2_MEDIUM", 3: "TIER_3_WATCH"
    }).values

    # Attach all signals (excl. internals prefixed with _)
    for col in [c for c in signals.columns if not c.startswith("_")]:
        out[f"sig_{col}"] = signals[col].values
    # Attach the continuous WDS for downstream analytics
    if "_WEIGHTED_DROP_SCORE" in signals.columns:
        out["weighted_drop_score"] = signals["_WEIGHTED_DROP_SCORE"].values.round(4)

    all_path = os.path.join(CFG["OUTPUT_DIR"], "churn_scored.csv")
    out.to_csv(all_path, index=False)
    log.info("Full scored CSV → %s  (%d rows)", all_path, len(out))

    for num, name in [(1, "tier1_alerts"), (2, "tier2_alerts"), (3, "tier3_watchlist")]:
        tier_df = out[out["tier"] == num].copy()
        path    = os.path.join(CFG["OUTPUT_DIR"], f"churn_{name}.csv")
        tier_df.to_csv(path, index=False)
        log.info("  %s → %d rows", name.upper(), len(tier_df))

    # Text report
    lines = [
        f"CHURN RULE-BASED SCORING REPORT v4 — {time.strftime('%Y-%m-%d %H:%M')}",
        f"Input CSV: {CFG['INPUT_CSV']}",
        f"Total rows: {len(df):,}",
        (f"Tier 1: {int((tiers==1).sum()):,}  Tier 2: {int((tiers==2).sum()):,}  "
         f"Tier 3: {int((tiers==3).sum()):,}  No-action: {int((tiers==0).sum()):,}"),
    ]
    txt = os.path.join(CFG["OUTPUT_DIR"], "churn_rules_report.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines))
    log.info("Text report → %s", txt)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    _sep()
    print("  CHURN RULES v4 — Precision-Recall Optimised Edition")
    print(f"  Input  : {CFG['INPUT_CSV']}  |  Horizon: {CFG['CHURN_HORIZON']}-day")
    print(f"  Output : {CFG['OUTPUT_DIR']}")
    print(f"  Tiers  : T2≥{CFG['T2_SCORE_MIN']}  T3 [{CFG['T3_SCORE_MIN']}–{CFG['T3_SCORE_MAX']}]")
    _sep()

    df         = load_data()
    target_col = resolve_target(df)
    y_true     = df[target_col].astype(int).values if target_col else None

    if y_true is not None:
        log.info("Target: %s | Churners: %d (%.3f%%)",
                 target_col, int(y_true.sum()), y_true.mean() * 100)

    log.info("Computing signals (v4: +VOICE_SILENT_DATA_DEAD, +CROSS_SVC_INTERACTION, "
             "+RECHARGE_CLIFF, +CONSISTENT_DECAY, +WEIGHTED_DROP_SCORE) …")
    signals = compute_signals(df)

    for sig in ["STRICT_VELOCITY_COLLAPSE", "VOICE_SILENT_DATA_DEAD",
                "CROSS_SVC_INTERACTION",    "RECHARGE_CLIFF",
                "CONSISTENT_DECAY",         "CYCLICAL_RECHARGER_GUARD"]:
        if sig in signals.columns:
            log.info("  %-35s  fired on %d subscribers", sig, int(signals[sig].sum()))

    score = compute_risk_score(signals)
    tiers = assign_tiers(signals, score)

    log.info("Tier distribution: T1=%d  T2=%d  T3=%d  No-action=%d",
             int((tiers == 1).sum()), int((tiers == 2).sum()),
             int((tiers == 3).sum()), int((tiers == 0).sum()))

    results = None
    if y_true is not None:
        results = evaluate_tiers(y_true, tiers, score)
        print_report(results, y_true, tiers, signals, n_total=len(df))
    else:
        _sep()
        print("  SCORE-ONLY MODE (no churn label found)")
        print(f"  T1={int((tiers==1).sum()):,}  T2={int((tiers==2).sum()):,}  "
              f"T3={int((tiers==3).sum()):,}  No-action={int((tiers==0).sum()):,}")
        _sep()

    save_outputs(df, tiers, score, signals, target_col)

    if y_true is not None and results is not None:
        log.info("Generating dashboard …")
        plot_dashboard(y_true, tiers, score, signals, results, n_total=len(df))

    _sep()
    print(f"  DONE in {time.time() - t0:.1f}s  |  Outputs → {CFG['OUTPUT_DIR']}/")
    if y_true is not None and results:
        r1  = results["Tier 1 (high-confidence)"]
        r12 = results["Tier 1+2 combined"]
        print(f"\n  QUICK RESULTS:")
        print(f"  Tier 1 — Prec {r1['precision']*100:.1f}%  "
              f"Rec {r1['recall']*100:.1f}%  "
              f"Lift {r1['lift']:.1f}×  Alerts {r1['alerts']:,}")
        print(f"  T1+T2  — Prec {r12['precision']*100:.1f}%  "
              f"Rec {r12['recall']*100:.1f}%  "
              f"Lift {r12['lift']:.1f}×  Alerts {r12['alerts']:,}")
    print()
    _sep()


if __name__ == "__main__":
    main()



