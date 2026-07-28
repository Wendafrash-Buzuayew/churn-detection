"""
churn_rules.py  ·  v3  (Velocity-Drop Edition)
================================================
Rule-Based Telecom Churn Detection Pipeline
───────────────────────────────────────────────────────────────────────────────

v3 changes — addressing false-positive flooding and feature masking:

  PROBLEM 1 — FP FLOODING (87.6% of Tier 1 FPs are stable low-frequency users)
    Root cause: original Tier 1 qualified ANY subscriber who looked "inactive"
    on 4-week aggregates, even if they were paying and voice-stable.
    Fix: STABLE_LOW_USER==0 is now a HARD GATE in Tier 1, not just a scorer.

  PROBLEM 2 — FN MASKING (80.8% of missed churners: 4W average looks healthy)
    Root cause: DATA_REVENUE_RECENT_4W and TOTAL_VOICE_ACTIVE_WEEKS act as
    safety drivers — their 4-week sum stays positive even when W13 collapsed.
    Fix: NEW `_velocity_drop()` function computes W13 vs W10–W12 *baseline*
    per service. STRICT_VELOCITY_COLLAPSE (≥2 services dropped ≥85%) is now
    an *alternative path into Tier 1*, catching churners the old AND rule missed.

  v3 Tier 1 rule (rewritten from scratch):
    SERVICE_COLLAPSED == 1
    AND NOT_RECOVERING == 1
    AND (STRICT_VELOCITY_COLLAPSE == 1 OR _ZERO_REV == 1)
    AND STABLE_LOW_USER == 0          ← hard suppressor, not just a penalty

OOT benchmark (from ML model):
  Best-threshold ML : Precision 2.9%  Recall 66.3%  Lift  4.9×  FP 3.48M
  Score≥0.95 ML     : Precision 18.3% Recall 22.1%  Lift 31.4×  FP 153K

Rule-based v2 results on sample data:
  TIER 1 (high-confidence) : Precision 61.1% Recall 22%  Lift 122×  Alerts ~6.7K
  TIER 2 (medium-risk)     : Precision  4.7% Recall 32%  Lift 9.4×  Alerts ~127K
  TIER 3 (watch-list)      : Precision  1.4% Recall 20%  Lift 2.8×  Alerts ~268K

Run:
    python churn_rules.py                                # uses Feb_Train.csv
    INPUT_CSV=your_file.csv python churn_rules.py        # any CSV
    INPUT_CSV=your_file.csv CHURN_HORIZON=30 python churn_rules.py

Output (./churn_rules_output/):
    churn_scored.csv            — every subscriber with tier + signals
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
    "INPUT_CSV"       : os.getenv("INPUT_CSV",      "Feb1_Train_with_recharg.csv"),
    "OUTPUT_DIR"      : os.getenv("OUTPUT_DIR",      "./churn_rules_output"),
    "ID_COL"          : os.getenv("ID_COL",          "MSISDN"),
    "CHURN_HORIZON"   : int(os.getenv("CHURN_HORIZON", "90")),
    "DATASET_TYPE_COL": "DATASET_TYPE",

    # ── TIER 1: rewritten in v3 ───────────────────────────────────────────────
    # Hard-AND gate: SERVICE_COLLAPSED AND NOT_RECOVERING
    #                AND (STRICT_VELOCITY_COLLAPSE OR ZERO_REV)
    #                AND NOT STABLE_LOW_USER
    # Rationale: STRICT_VELOCITY_COLLAPSE replaces the old NEARLY_INACTIVE
    # condition. A subscriber need not look "inactive" on 4W aggregates to
    # be a churner — they just need a demonstrable W13 collapse across ≥2
    # services relative to their W10–W12 baseline.
    "T1_RULES": {
        "VELOCITY_DROP_THRESHOLD"    : 0.85,  # ≥85% drop from W10-W12 baseline
        "VELOCITY_COLLAPSE_MIN_SVC"  : 2,     # ≥2 services must drop for STRICT flag
        "STABLE_LOW_USER_HARD_BLOCK" : True,  # STABLE_LOW_USER==1 → never Tier 1
    },

    # ── TIER 2: medium-risk score threshold ───────────────────────────────────
    # Kept at 9 (raised from 8 in v2 to accommodate new positive weights).
    "T2_SCORE_MIN": 9,

    # ── TIER 3: watch-list ────────────────────────────────────────────────────
    "T3_SCORE_MIN": 6,
    "T3_SCORE_MAX": 8,

    # ── Signal weights (SHAP-calibrated, v3 additions documented below) ───────
    #
    # v3 NEW signals:
    #   DATA_VELOCITY_DROP    +1.5  individual service velocity flag (≥85% drop)
    #   VOICE_VELOCITY_DROP   +1.5  individual service velocity flag
    #   BUNDLE_VELOCITY_DROP  +1.5  individual service velocity flag
    #   STRICT_VELOCITY_COLLAPSE +3.0  composite: ≥2 services with ≥85% drop
    #
    # Existing weights from v2 retained (all SHAP-justified, see v2 docstring):
    "RULE_WEIGHTS": {
        # ── Primary churn risk signals (2+ pts) ──────────────────────────────
        "SERVICE_COLLAPSED"          : 2.5,
        "NOT_RECOVERING"             : 2.0,
        "NEAR_ZERO_REVENUE"          : 2.0,
        "ALL_SERVICES_ZERO_W13"      : 2.0,
        "W13_VELOCITY_COLLAPSE"      : 2.5,   # ≥2 services dropped >80% (v2 signal)

        # ── v3 NEW: per-service acceleration collapse flags ───────────────────
        # Individual flags: 1.5 pts each. When all three fire simultaneously,
        # combined score contribution is 4.5 pts, correctly dominating over
        # the 4W aggregate safety features. STRICT_VELOCITY_COLLAPSE (the
        # composite) adds an additional 3.0 pts on top.
        "DATA_VELOCITY_DROP"         : 1.5,   # NEW v3: data MB W13 drop ≥85%
        "VOICE_VELOCITY_DROP"        : 1.5,   # NEW v3: voice min W13 drop ≥85%
        "BUNDLE_VELOCITY_DROP"       : 1.5,   # NEW v3: bundle cnt W13 drop ≥85%
        "STRICT_VELOCITY_COLLAPSE"   : 3.0,   # NEW v3: ≥2 services with ≥85% drop

        # ── Medium signals (1.0–1.5 pts) ─────────────────────────────────────
        "NEARLY_INACTIVE_RECENT"     : 1.5,
        "SERVICE_DIVERSITY_DROPPING" : 1.5,
        "ACTIVITY_TRENDING_DOWN"     : 1.0,
        "REVENUE_UNDER_50"           : 1.0,
        "NO_BUNDLE_RECENT"           : 1.0,

        # ── Lower weight signals ──────────────────────────────────────────────
        "LOW_DATA_ACTIVITY"          : 0.5,
        "NO_VOICE_ACTIVITY"          : 0.5,

        # ── False-positive suppressors (negative weights) ─────────────────────
        "RECOVERING_MULTI_SERVICE"   : -2.0,
        "STABLE_LOW_USER"            : -2.5,  # PRIMARY FP suppressor (79.8% hit)
        "BUNDLE_WITH_REVENUE"        : -1.5,
        "VOICE_STABLE_REVENUE"       : -1.0,
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
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    """Safely fetch a numeric column; zero-fill if absent."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default).values
    return np.full(len(df), default, dtype=np.float32)


def _weekly_matrix(df: pd.DataFrame, prefix: str,
                   weeks=("W10", "W11", "W12", "W13")) -> np.ndarray:
    """
    Return an (n, 4) float32 matrix for [W10, W11, W12, W13] of `prefix`.
    Missing columns are zero-filled so downstream code never breaks on
    partial schemas.
    """
    cols  = [f"{prefix}_{w}" for w in weeks]
    frame = (
        df.reindex(columns=cols, fill_value=0.0)
          .apply(pd.to_numeric, errors="coerce")
          .fillna(0.0)
    )
    return frame.values.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# VELOCITY DROP  (v3 core addition)
# ─────────────────────────────────────────────────────────────────────────────

def _velocity_drop(mat: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    """
    Compute per-subscriber W13 acceleration collapse for a single service.

    Algorithm
    ─────────
    1. Compute the W10–W12 baseline average (3-week mean before the final week).
    2. Compute drop percentage: (baseline − W13) / baseline.
       Division-by-zero is handled explicitly: if baseline == 0 the subscriber
       had no prior activity, so the signal is NOT fired (drop_pct = 0).
       This prevents false-positive firing on subscribers who were already dead
       in W10–W12 and simply stayed dead in W13 — those are handled by
       ALL_SERVICES_ZERO_W13 instead.
    3. Return a boolean (0/1) array: True where drop_pct >= threshold.

    Parameters
    ──────────
    mat       : (n, 4) float32 matrix of [W10, W11, W12, W13] for one service.
    threshold : minimum fractional drop to flag. Default 0.85 (≥85% collapse).

    Returns
    ───────
    np.ndarray of int8, shape (n,).  1 = velocity collapsed, 0 = did not.
    """
    baseline = mat[:, :3].mean(axis=1)   # mean(W10, W11, W12) per subscriber
    w13      = mat[:, 3]                  # final week value

    # np.where prevents division by zero: where baseline == 0, drop_pct = 0
    drop_pct = np.where(
        baseline > 0.0,
        (baseline - w13) / baseline,     # positive = W13 below baseline
        0.0,
    )
    return (drop_pct >= threshold).astype(np.int8)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all named churn signals.  Returns a DataFrame of 0/1 integer columns.

    v3 additions:
      DATA_VELOCITY_DROP      — data MB W13 collapsed ≥85% vs W10-W12 mean
      VOICE_VELOCITY_DROP     — voice min W13 collapsed ≥85% vs W10-W12 mean
      BUNDLE_VELOCITY_DROP    — bundle cnt W13 collapsed ≥85% vs W10-W12 mean
      STRICT_VELOCITY_COLLAPSE— composite: ≥2 of the above fire simultaneously
        → this is the primary v3 FN-reducer and new Tier 1 entry path

    All v2 signals are retained unchanged.  The STABLE_LOW_USER suppressor
    is now also enforced as a hard gate in assign_tiers() (not just a score
    penalty), per the v3 design requirement.
    """
    S = {}
    VD_THRESHOLD = CFG["T1_RULES"]["VELOCITY_DROP_THRESHOLD"]   # default 0.85

    # ── Weekly usage matrices (n × 4: columns = W10, W11, W12, W13) ──────────
    data_mat   = _weekly_matrix(df, "DATA_MB")
    voice_mat  = _weekly_matrix(df, "OG_VOICE_MIN")
    bundle_mat = _weekly_matrix(df, "BUNDLE_CNT")
    sms_mat    = _weekly_matrix(df, "OG_SMS_COUNT")

    # ── Recovery guard: is W13 > W10 for each service? ───────────────────────
    # Used by NOT_RECOVERING and RECOVERING_MULTI_SERVICE.
    data_recovering   = (data_mat[:,   -1] > data_mat[:,   0]).astype(np.int8)
    voice_recovering  = (voice_mat[:,  -1] > voice_mat[:,  0]).astype(np.int8)
    bundle_recovering = (bundle_mat[:, -1] > bundle_mat[:, 0]).astype(np.int8)
    recovery_count    = data_recovering + voice_recovering + bundle_recovering

    # ── Revenue columns ───────────────────────────────────────────────────────
    total_rev    = _get_col(df, "TOTAL_REVENUE_RECENT_4W")
    data_rev     = _get_col(df, "DATA_REVENUE_RECENT_4W")
    voice_active = _get_col(df, "TOTAL_VOICE_ACTIVE_WEEKS_RECENT_4W")

    # ═════════════════════════════════════════════════════════════════════════
    # CHURN RISK SIGNALS
    # ═════════════════════════════════════════════════════════════════════════

    # ── Signal 1: Service diversity collapsed (≤1 active service) ────────────
    # High-confidence, not in top FP drivers.  Weight: 2.5.
    S["SERVICE_COLLAPSED"] = (
        _get_col(df, "SERVICE_DIVERSITY_RECENT_4W") <= 1
    ).astype(np.int8)

    # ── Signal 2: No service bouncing back ───────────────────────────────────
    # Permanent drops characterise churners; non-churners recover.
    S["NOT_RECOVERING"] = (recovery_count == 0).astype(np.int8)

    # ── Signal 3: Near-zero revenue (≤10 total revenue in 4W) ────────────────
    # FPs have positive revenue; this only fires for genuine zero-spend subs.
    S["NEAR_ZERO_REVENUE"] = (total_rev <= 10).astype(np.int8)

    # ── Signal 4: All services completely dead in W13 ─────────────────────────
    # Strongest single-week dead-SIM indicator.
    S["ALL_SERVICES_ZERO_W13"] = (
        (data_mat[:, -1]   <= 0) &
        (voice_mat[:, -1]  <= 0) &
        (bundle_mat[:, -1] <= 0)
    ).astype(np.int8)

    # ── Signal 5: W13 velocity collapse (v2 — 80% threshold, ≥2 services) ───
    # Catches ~54% of FNs masked by 4W aggregates. This is the softer version
    # (80% drop on ≥2 services). The stricter 85% version is Signal 6.
    def _w13_collapse_80(mat):
        avg_w10_w12 = mat[:, :3].mean(axis=1)
        w13         = mat[:, 3]
        drop_ratio  = (avg_w10_w12 - w13) / (avg_w10_w12 + 1.0)
        return ((drop_ratio >= 0.80) & (avg_w10_w12 > 0)).astype(np.int8)

    data_collapse_80   = _w13_collapse_80(data_mat)
    voice_collapse_80  = _w13_collapse_80(voice_mat)
    bundle_collapse_80 = _w13_collapse_80(bundle_mat)
    S["W13_VELOCITY_COLLAPSE"] = (
        (data_collapse_80 + voice_collapse_80 + bundle_collapse_80) >= 2
    ).astype(np.int8)

    # ── Signal 6 (NEW v3): Per-service velocity drop flags (≥85% threshold) ──
    #
    # WHY A SEPARATE 85% THRESHOLD (vs the 80% in Signal 5):
    #   The 80% W13_VELOCITY_COLLAPSE is a score signal — it fires broadly to
    #   boost Tier 2 recall.  The 85% per-service flags are designed for the
    #   new Tier 1 hard-gate: we want higher confidence (stricter drop) and
    #   individual-service granularity so the Tier 1 rule can reference specific
    #   service failures rather than only a multi-service composite.
    #
    # FORMULA (per _velocity_drop docstring):
    #   baseline = mean(W10, W11, W12)
    #   drop_pct = (baseline − W13) / baseline   [safe: 0 when baseline == 0]
    #   flag = 1 if drop_pct >= 0.85
    #
    # ZERO-ACTIVITY GUARD: _velocity_drop only fires when baseline > 0, so
    # subscribers who were already dead in W10–W12 do NOT trigger this signal.
    # They are captured by ALL_SERVICES_ZERO_W13 instead.
    data_vdrop   = _velocity_drop(data_mat,   threshold=VD_THRESHOLD)
    voice_vdrop  = _velocity_drop(voice_mat,  threshold=VD_THRESHOLD)
    bundle_vdrop = _velocity_drop(bundle_mat, threshold=VD_THRESHOLD)

    S["DATA_VELOCITY_DROP"]   = data_vdrop    # data MB: W13 < 15% of W10-W12 avg
    S["VOICE_VELOCITY_DROP"]  = voice_vdrop   # voice min: W13 < 15% of W10-W12 avg
    S["BUNDLE_VELOCITY_DROP"] = bundle_vdrop  # bundle cnt: W13 < 15% of W10-W12 avg

    # ── Signal 7 (NEW v3): STRICT_VELOCITY_COLLAPSE ───────────────────────────
    # Composite: at least VELOCITY_COLLAPSE_MIN_SVC (default 2) services show
    # a ≥85% drop in W13 vs their W10–W12 baseline.
    #
    # DESIGN RATIONALE:
    #   Requiring ≥2 services limits single-service noise (e.g. a voice-only
    #   subscriber who simply stopped calling in one week). When ≥2 services
    #   simultaneously collapse, the probability that this is a real churn event
    #   is much higher — multi-service simultaneous drops are rare in non-churners.
    #
    # TIER 1 ROLE:
    #   This flag serves as the *alternative path into Tier 1* for churners
    #   who were previously missed because their 4W aggregates looked healthy.
    #   A subscriber can now reach Tier 1 either via the traditional
    #   ZERO_REV path (dead-SIM with no revenue) OR via STRICT_VELOCITY_COLLAPSE
    #   (sudden terminal collapse in W13 even with a positive prior baseline).
    n_vdrops_min = CFG["T1_RULES"]["VELOCITY_COLLAPSE_MIN_SVC"]
    S["STRICT_VELOCITY_COLLAPSE"] = (
        (data_vdrop + voice_vdrop + bundle_vdrop) >= n_vdrops_min
    ).astype(np.int8)

    # ── Signal 8: Nearly inactive in recent 4 weeks ───────────────────────────
    S["NEARLY_INACTIVE_RECENT"] = (
        _get_col(df, "ANY_ACTIVE_WEEKS_RECENT_4W") <= 1
    ).astype(np.int8)

    # ── Signal 9: Service diversity trending downward ─────────────────────────
    S["SERVICE_DIVERSITY_DROPPING"] = (
        _get_col(df, "SERVICE_DIVERSITY_DROP") == 1
    ).astype(np.int8)

    # ── Signal 10: Activity trending downward ─────────────────────────────────
    S["ACTIVITY_TRENDING_DOWN"] = (
        _get_col(df, "ANY_ACTIVE_WEEKS_DROP") == 1
    ).astype(np.int8)

    # ── Signal 11: Revenue under 50 ──────────────────────────────────────────
    S["REVENUE_UNDER_50"] = (total_rev <= 50).astype(np.int8)

    # ── Signal 12: No bundle purchases in 4 weeks ─────────────────────────────
    S["NO_BUNDLE_RECENT"] = (
        _get_col(df, "BUNDLE_ACTIVE_WEEKS_RECENT_4W") == 0
    ).astype(np.int8)

    # ── Signal 13: Low data activity ──────────────────────────────────────────
    S["LOW_DATA_ACTIVITY"] = (
        _get_col(df, "DATA_ACTIVE_WEEKS_RECENT_4W") <= 1
    ).astype(np.int8)

    # ── Signal 14: No voice activity ──────────────────────────────────────────
    S["NO_VOICE_ACTIVITY"] = (voice_active == 0).astype(np.int8)

    # ═════════════════════════════════════════════════════════════════════════
    # FALSE-POSITIVE SUPPRESSORS  (negative weights — penalise safe users)
    # ═════════════════════════════════════════════════════════════════════════

    # ── Suppressor 1: Multi-service recovery (validated 0% FN impact) ─────────
    S["RECOVERING_MULTI_SERVICE"] = (recovery_count >= 2).astype(np.int8)

    # ── Suppressor 2: Stable low-frequency user (PRIMARY FP suppressor) ───────
    # SHAP PATTERN A: 79.8% of FPs have positive data revenue AND stable voice.
    # These are LOW-FREQUENCY but PAYING users — not churners.
    #
    # v3 CHANGE: STABLE_LOW_USER is now ALSO enforced as a hard gate in
    # assign_tiers() (in addition to the −2.5 score penalty here).
    # This ensures no STABLE_LOW_USER can ever be assigned Tier 1 regardless
    # of how many other signals fire.
    S["STABLE_LOW_USER"] = (
        (data_rev > 0) &          # generating some data revenue
        (voice_active >= 2)        # voice still active in ≥2 of the last 4 weeks
    ).astype(np.int8)

    # ── Suppressor 3: Bundle purchase with positive revenue ───────────────────
    # SHAP PATTERN B: 69.8% of FPs bought bundles in W13 AND have positive rev.
    S["BUNDLE_WITH_REVENUE"] = (
        (bundle_mat[:, -1] > 0) &  # bundle bought in W13
        (total_rev > 0)             # any 4W revenue (not purely zero-spend)
    ).astype(np.int8)

    # ── Suppressor 4: Voice-stable with positive revenue ──────────────────────
    # SHAP PATTERN C: 70.7% of FPs have voice-active weeks ≥2 AND revenue > 0.
    S["VOICE_STABLE_REVENUE"] = (
        (voice_active >= 2) &
        (total_rev > 0)
    ).astype(np.int8)

    # ── AUXILIARY COLUMNS (Tier 1 hard rule — not scored) ─────────────────────
    S["_ZERO_REV"]           = (total_rev <= 0).astype(np.int8)
    S["_DEAD_W13"]           = S["ALL_SERVICES_ZERO_W13"]
    S["_RECOVERY_COUNT"]     = recovery_count
    S["_W13_COLLAPSE_COUNT"] = (data_collapse_80 + voice_collapse_80 + bundle_collapse_80)
    # v3 additions — stored for diagnostics / downstream reporting
    S["_N_VDROP_SERVICES"]   = (data_vdrop + voice_vdrop + bundle_vdrop)

    return pd.DataFrame(S, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# RISK SCORING
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_score(signals: pd.DataFrame) -> np.ndarray:
    """
    Weighted sum of all signals.
    Positive score = higher churn risk.
    Negative score = strong recovery / revenue behaviour (likely FP).
    """
    weights = CFG["RULE_WEIGHTS"]
    score   = np.zeros(len(signals), dtype=np.float32)
    for signal_name, weight in weights.items():
        if signal_name in signals.columns:
            score += weight * signals[signal_name].values
    return score


# ─────────────────────────────────────────────────────────────────────────────
# TIER ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def assign_tiers(signals: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """
    Assign each subscriber to a tier (0 = no risk, 1 = highest risk).

    ──────────────────────────────────────────────────────────────────────────
    TIER 1 — High-confidence churn risk  (v3 hard-AND gate, fully rewritten)
    ──────────────────────────────────────────────────────────────────────────
    Condition (ALL must be true):

      [A] SERVICE_COLLAPSED == 1
            Service diversity has collapsed to ≤1 service in recent 4 weeks.
            This is the structural backbone of the churn profile.

      [B] NOT_RECOVERING == 1
            No service shows W13 > W10.  Drops are permanent, not cyclical.

      [C] STRICT_VELOCITY_COLLAPSE == 1  OR  _ZERO_REV == 1
            Either path suffices:
            • STRICT_VELOCITY_COLLAPSE: ≥2 services each dropped ≥85% from
              their W10–W12 mean in W13.  Catches churners whose 4W aggregate
              still looks healthy but whose W13 is already terminal.
            • _ZERO_REV: zero total revenue in the 4-week window.
              The original dead-SIM gate — kept as an alternative entry point.

      [D] STABLE_LOW_USER == 0  (HARD BLOCK — v3 change)
            A subscriber with positive data revenue AND voice-active ≥2 weeks
            is a low-frequency paying user, not a churner.  In v2 this was
            only a score penalty (−2.5).  In v3 it is a hard exclusion from
            Tier 1, eliminating the 87.6% FP pattern at the gate level.

    ──────────────────────────────────────────────────────────────────────────
    TIER 2 — Medium-risk  (risk score ≥ T2_SCORE_MIN, not already Tier 1)
    ──────────────────────────────────────────────────────────────────────────

    TIER 3 — Watch-list  (score in [T3_SCORE_MIN, T3_SCORE_MAX])

    TIER 0 — No action
    """

    # ── Gate [A]: service diversity collapsed ─────────────────────────────────
    svc_collapsed = signals["SERVICE_COLLAPSED"].values == 1

    # ── Gate [B]: no recovery in any service ─────────────────────────────────
    not_recovering = signals["NOT_RECOVERING"].values == 1

    # ── Gate [C]: velocity collapse OR zero revenue ───────────────────────────
    # STRICT_VELOCITY_COLLAPSE: ≥2 services with ≥85% W13 drop (v3 new path)
    strict_vel_collapse = signals["STRICT_VELOCITY_COLLAPSE"].values == 1
    zero_rev            = signals["_ZERO_REV"].values == 1
    churn_signal_gate   = strict_vel_collapse | zero_rev

    # ── Gate [D]: hard FP suppressor — must NOT be a stable low user ─────────
    # This is the critical v3 addition: STABLE_LOW_USER is evaluated as a
    # binary exclusion BEFORE the score is consulted, preventing the 87.6%
    # FP pattern from ever reaching Tier 1 regardless of score.
    not_stable_low_user = signals["STABLE_LOW_USER"].values == 0

    # ── Tier 1 combined gate ──────────────────────────────────────────────────
    tier1_mask = (
        svc_collapsed       &   # [A]
        not_recovering      &   # [B]
        churn_signal_gate   &   # [C]
        not_stable_low_user     # [D]
    )

    # ── Tier 2: score-driven ──────────────────────────────────────────────────
    tier2_mask = (score >= CFG["T2_SCORE_MIN"]) & (~tier1_mask)

    # ── Tier 3: watch-list band ───────────────────────────────────────────────
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
        ("Tier 1+2 combined",         ((tiers >= 1) & (tiers <= 2)).astype(int)),
        ("Tier 1+2+3 combined",       (tiers >= 1).astype(int)),
        ("Tier 2 only",               (tiers == 2).astype(int)),
        ("Tier 3 only",               (tiers == 3).astype(int)),
    ]:
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
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
    print(f"  Churners caught  : {tp}/{pos} = {tp / max(pos, 1) * 100:.1f}%")
    print(f"  Churners missed  : {fn}/{pos} = {fn / max(pos, 1) * 100:.1f}%")
    print(f"  False alarms     : {fp:,}")
    print(f"  Alert precision  : {tp / max(tp + fp, 1) * 100:.1f}%")
    _sep("─", 62)
    print()


def print_report(results: Dict, y_true: np.ndarray, tiers: np.ndarray,
                 signals: pd.DataFrame, n_total: int):
    base_rate = y_true.mean() * 100 if y_true is not None else 0

    _sep()
    print("  RULE-BASED CHURN DETECTION — EVALUATION REPORT  [v3 Velocity-Drop]")
    _sep()
    print(f"  Total subscribers : {n_total:,}")
    if y_true is not None:
        print(f"  Actual churners   : {int(y_true.sum()):,}  ({base_rate:.3f}%)")
    print()

    # Per-tier summary table
    print("  ── TIER PERFORMANCE SUMMARY ──")
    print(f"  {'Segment':<28}  {'Alerts':>8}  {'TP':>6}  {'FP':>8}  "
          f"{'Precision':>10}  {'Recall':>8}  {'Lift':>8}")
    print("  " + "─" * 78)
    for label, r in results.items():
        print(f"  {label:<28}  {r['alerts']:>8,}  {r['tp']:>6}  {r['fp']:>8,}  "
              f"  {r['precision'] * 100:>8.2f}%  {r['recall'] * 100:>7.2f}%  "
              f"{r['lift']:>7.1f}x")
    print()

    # Confusion matrices
    r1  = results["Tier 1 (high-confidence)"]
    r12 = results["Tier 1+2 combined"]
    _print_cm(r1["tp"],  r1["fp"],  r1["fn"],  r1["tn"],  "TIER 1 — High-confidence")
    _print_cm(r12["tp"], r12["fp"], r12["fn"], r12["tn"], "TIER 1+2 — Campaign base")

    # Signal diagnostics — separate new v3 signals for easy audit
    _sep("─", 62)
    print("  NEW v3 SIGNAL DIAGNOSTICS  (velocity-drop signals)")
    _sep("─", 62)
    v3_sigs = ["DATA_VELOCITY_DROP", "VOICE_VELOCITY_DROP",
               "BUNDLE_VELOCITY_DROP", "STRICT_VELOCITY_COLLAPSE"]
    for col in v3_sigs:
        if col not in signals.columns:
            continue
        n_fired = int(signals[col].sum())
        if y_true is not None and n_fired:
            tp_s    = int(((signals[col] == 1) & (y_true == 1)).sum())
            prec_s  = tp_s / n_fired
            weight  = CFG["RULE_WEIGHTS"].get(col, 0)
            print(f"  {col:<35}  fired={n_fired:>6,}  TP={tp_s:>4}  "
                  f"prec={prec_s:.4f}  weight={weight:+.1f}")
    print()

    # Full signal firing table
    _sep("─", 62)
    print("  ALL SIGNAL FIRING RATES")
    _sep("─", 62)
    signal_cols = [c for c in signals.columns if not c.startswith("_")]
    for col in signal_cols:
        n_fired = int(signals[col].sum())
        if y_true is not None:
            tp_s   = int(((signals[col] == 1) & (y_true == 1)).sum())
            prec_s = tp_s / n_fired if n_fired else 0
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
    for label, r in results.items():
        if "combined" in label or "only" in label:
            continue
        proj_alerts = int(r["alerts"] * scale)
        print(f"\n  {label}")
        print(f"    Daily alerts       : {proj_alerts:>10,}")
        print(f"    Projected precision: {r['precision'] * 100:>9.2f}%")
        print(f"    Projected recall   : {r['recall'] * 100:>9.2f}%")
        print(f"    Lift vs random     : {r['lift']:>9.1f}×")
    print()

    # Strategy
    _sep()
    print("  RECOMMENDED CAMPAIGN STRATEGY  [v3]")
    _sep()
    print("""
  TIER 1 — CRM / Retention Specialist Team
  ──────────────────────────────────────────
  Rule : SERVICE_COLLAPSED AND NOT_RECOVERING
         AND (STRICT_VELOCITY_COLLAPSE OR ZERO_REVENUE)
         AND NOT STABLE_LOW_USER
  v3 change: STRICT_VELOCITY_COLLAPSE opens a new entry path for churners
  whose 4W aggregates still looked healthy. STABLE_LOW_USER is now a hard
  block (not just a penalty) eliminating 87.6% of v2's FP pattern entirely.

  TIER 2 — Automated Retention Campaign
  ──────────────────────────────────────
  Rule : Risk score ≥ 9 (not already in Tier 1)
  Best for: automated SMS, data bundle offer, targeted CVM push notification.

  TIER 3 — Low-Cost Digital Watch List
  ──────────────────────────────────────
  Rule : Risk score 6–8
  Best for: app push notification, email, recharge incentive.
""")
    _sep()


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(y_true, tiers, score, signals, results, n_total):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 13))
    fig.suptitle("Rule-Based Churn Detection v3 — Velocity-Drop Dashboard",
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
        "safe": "#27AE60", "accent": "#2980B9",
    }

    # ── Panel 1: Tier funnel ──────────────────────────────────────────────────
    labels = ["Tier 1\nHigh-conf.", "Tier 2\nMedium", "Tier 3\nWatch", "No Action"]
    colors = [COLORS["tier1"], COLORS["tier2"], COLORS["tier3"], COLORS["safe"]]
    counts = [int((tiers == i).sum()) for i in [1, 2, 3, 0]]
    bars   = ax1.bar(labels, counts, color=colors, edgecolor="white", width=0.6)
    ax1.set_title("Subscriber Distribution by Tier", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Subscriber Count", fontsize=9)
    for bar, cnt in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{cnt:,}", ha="center", fontsize=8.5, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: Precision vs Recall by tier ──────────────────────────────────
    if y_true is not None:
        tier_labels  = ["Tier 1", "Tier 2", "Tier 3"]
        tier_precs   = [results["Tier 1 (high-confidence)"]["precision"],
                        results["Tier 2 only"]["precision"],
                        results["Tier 3 only"]["precision"]]
        tier_recalls = [results["Tier 1 (high-confidence)"]["recall"],
                        results["Tier 2 only"]["recall"],
                        results["Tier 3 only"]["recall"]]
        tier_colors  = [COLORS["tier1"], COLORS["tier2"], COLORS["tier3"]]
        for lbl, p, r, c in zip(tier_labels, tier_precs, tier_recalls, tier_colors):
            ax2.scatter(r, p, color=c, s=300, zorder=5, label=lbl)
            ax2.annotate(lbl, (r, p), textcoords="offset points", xytext=(8, 4), fontsize=9)
        ax2.scatter(0.663, 0.029, marker="X", color="#7F8C8D", s=200, zorder=4,
                    label="ML best-th (OOT)")
        ax2.scatter(0.221, 0.183, marker="X", color="#2C3E50", s=200, zorder=4,
                    label="ML score≥0.95 (OOT)")
        ax2.set_title("Precision vs Recall\n(vs ML OOT benchmark)", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Recall", fontsize=9); ax2.set_ylabel("Precision", fontsize=9)
        ax2.legend(fontsize=8); ax2.spines[["top", "right"]].set_visible(False)
        ax2.set_xlim(-0.05, 1.0); ax2.set_ylim(-0.05, 0.8)

    # ── Panel 3: Lift comparison ───────────────────────────────────────────────
    if y_true is not None:
        lift_labels = ["ML\nbest-th\n(OOT)", "ML\n≥0.95\n(OOT)",
                       "Rules\nTier 1+2", "Rules\nTier 1"]
        lift_vals   = [4.94, 31.44,
                       results["Tier 1+2 combined"]["lift"],
                       results["Tier 1 (high-confidence)"]["lift"]]
        bar_colors  = ["#7F8C8D", "#7F8C8D", COLORS["tier2"], COLORS["tier1"]]
        bars = ax3.bar(lift_labels, lift_vals, color=bar_colors, edgecolor="white", width=0.55)
        for bar, val in zip(bars, lift_vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}×", ha="center", fontsize=9, fontweight="bold")
        ax3.set_title("Lift vs Random Selection", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Lift (×)", fontsize=9)
        ax3.spines[["top", "right"]].set_visible(False)

    # ── Panel 4: Risk score distribution ─────────────────────────────────────
    if y_true is not None:
        score_churn = score[y_true == 1]
        score_nonch = score[y_true == 0]
        bins = np.linspace(score.min(), score.max(), 35)
        ax4.hist(score_nonch, bins=bins, alpha=0.65, color=COLORS["safe"],
                 label="Non-Churn", density=True)
        ax4.hist(score_churn, bins=bins, alpha=0.85, color=COLORS["tier1"],
                 label="Churn", density=True)
        ax4.axvline(CFG["T2_SCORE_MIN"], color="black", ls="-.", lw=1.5,
                    label=f"T2 cut (≥{CFG['T2_SCORE_MIN']})")
        ax4.axvline(CFG["T3_SCORE_MIN"], color="gray", ls=":", lw=1.2,
                    label=f"T3 cut (≥{CFG['T3_SCORE_MIN']})")
        ax4.set_title("Risk Score Distribution", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Rule-Based Risk Score"); ax4.set_ylabel("Density")
        ax4.legend(fontsize=8.5); ax4.spines[["top", "right"]].set_visible(False)

    # ── Panel 5: Tier 1 confusion matrix heatmap ─────────────────────────────
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
            f"Precision={r1['precision'] * 100:.1f}%  "
            f"Recall={r1['recall'] * 100:.1f}%  "
            f"Lift={r1['lift']:.1f}×",
            fontsize=10, fontweight="bold")
        ax5.tick_params(labelsize=8)

    # ── Panel 6: Signal churn-rate bar chart ──────────────────────────────────
    if y_true is not None:
        sig_cols = [c for c in signals.columns if not c.startswith("_")]
        rates    = {}
        for c in sig_cols:
            n_fired = int(signals[c].sum())
            if n_fired == 0:
                rates[c] = {"churn_rate": 0, "fire_rate": 0, "weight": 0}
            else:
                tp_s = int(((signals[c] == 1) & (y_true == 1)).sum())
                rates[c] = {
                    "churn_rate": round(tp_s / n_fired * 100, 1),
                    "fire_rate" : round(n_fired / len(y_true) * 100, 1),
                    "weight"    : CFG["RULE_WEIGHTS"].get(c, 0),
                }
        sig_df = pd.DataFrame(rates).T.sort_values("churn_rate", ascending=False)
        x      = np.arange(len(sig_df))
        c1     = [COLORS["tier1"] if w > 0 else COLORS["safe"] for w in sig_df["weight"]]
        ax6.bar(x, sig_df["churn_rate"], color=c1, edgecolor="white")
        ax6.set_xticks(x)
        ax6.set_xticklabels(
            [s.replace("_", " ").title()[:18] for s in sig_df.index],
            rotation=45, ha="right", fontsize=7)
        ax6.set_title("Churn Rate When Signal Fires (%)", fontsize=11, fontweight="bold")
        ax6.set_ylabel("% of flagged = actual churner", fontsize=9)
        ax6.axhline(y_true.mean() * 100, color="black", ls="--", lw=1.2,
                    label=f"Base rate ({y_true.mean() * 100:.2f}%)")
        ax6.legend(fontsize=8.5); ax6.spines[["top", "right"]].set_visible(False)

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

    scored = df[[id_col]].copy() if id_col in df.columns else pd.DataFrame(index=df.index)
    if target_col and target_col in df.columns:
        scored["actual_churn"] = df[target_col].values

    scored["risk_score"]   = score.round(2)
    scored["tier"]         = tiers
    scored["tier_label"]   = pd.Series(tiers).map({
        0: "NO_ACTION", 1: "TIER_1_HIGH", 2: "TIER_2_MEDIUM", 3: "TIER_3_WATCH"
    }).values

    # Attach key signals for explainability in downstream CRM systems
    signal_cols = [c for c in signals.columns if not c.startswith("_")]
    for col in signal_cols:
        scored[f"sig_{col}"] = signals[col].values

    # Full scored population
    all_path = os.path.join(CFG["OUTPUT_DIR"], "churn_scored.csv")
    scored.to_csv(all_path, index=False)
    log.info("Full scored CSV → %s  (%d rows)", all_path, len(scored))

    # Per-tier alert files
    for tier_num, tier_name in [(1, "tier1_alerts"), (2, "tier2_alerts"), (3, "tier3_watchlist")]:
        tier_df = scored[scored["tier"] == tier_num].copy()
        path    = os.path.join(CFG["OUTPUT_DIR"], f"churn_{tier_name}.csv")
        tier_df.to_csv(path, index=False)
        log.info("  %s → %d rows → %s", tier_name.upper(), len(tier_df), path)

    # Text report summary
    report_lines = [
        f"CHURN RULE-BASED SCORING REPORT v3 — {time.strftime('%Y-%m-%d %H:%M')}",
        f"Input CSV: {CFG['INPUT_CSV']}",
        f"Total rows: {len(df):,}",
        (f"Tier 1: {int((tiers==1).sum()):,}  Tier 2: {int((tiers==2).sum()):,}  "
         f"Tier 3: {int((tiers==3).sum()):,}  No-action: {int((tiers==0).sum()):,}"),
    ]
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
    print("  RULE-BASED CHURN DETECTION PIPELINE  [v3 — Velocity-Drop Edition]")
    print(f"  Input: {CFG['INPUT_CSV']}  |  Horizon: {CFG['CHURN_HORIZON']}-day")
    print(f"  Output: {CFG['OUTPUT_DIR']}")
    _sep()

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_data()
    target_col = resolve_target(df)
    y_true = df[target_col].astype(int).values if target_col else None

    if y_true is not None:
        log.info("Target: %s | Churners: %d (%.3f%%)",
                 target_col, int(y_true.sum()), y_true.mean() * 100)

    # ── Compute signals ───────────────────────────────────────────────────────
    log.info("Computing churn signals (v3 — with velocity-drop) …")
    signals = compute_signals(df)

    # Log v3-specific signal firing counts for quick sanity check
    for sig in ["DATA_VELOCITY_DROP", "VOICE_VELOCITY_DROP",
                "BUNDLE_VELOCITY_DROP", "STRICT_VELOCITY_COLLAPSE"]:
        if sig in signals.columns:
            log.info("  %s fired on %d subscribers", sig, int(signals[sig].sum()))

    # ── Score and tier ────────────────────────────────────────────────────────
    log.info("Scoring and assigning tiers …")
    score = compute_risk_score(signals)
    tiers = assign_tiers(signals, score)

    log.info("Tier distribution: T1=%d  T2=%d  T3=%d  No-action=%d",
             int((tiers == 1).sum()), int((tiers == 2).sum()),
             int((tiers == 3).sum()), int((tiers == 0).sum()))

    # ── Evaluate ──────────────────────────────────────────────────────────────
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
    elapsed = time.time() - t0
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Outputs written to: {CFG['OUTPUT_DIR']}/")
    print(f"    churn_scored.csv          — every subscriber with tier + signals")
    print(f"    churn_tier1_alerts.csv    — high-confidence campaign list")
    print(f"    churn_tier2_alerts.csv    — medium-risk automated campaign")
    print(f"    churn_tier3_watchlist.csv — digital/low-cost channel")
    print(f"    churn_rules_dashboard.png — visual summary")
    _sep()

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
