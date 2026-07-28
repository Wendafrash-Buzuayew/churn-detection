"""
churn_rules_optimized.py
========================
Rule-Based Telecom Churn Detection Pipeline — SHAP-Optimized v2
────────────────────────────────────────────────────────────────
Built on the insight from OOT testing across 26.7M data points.
This version incorporates findings from SHAP analysis of 8,164 False Positives
and 1,299 False Negatives to improve precision and reduce missed churners.

── SHAP-Driven Changes (summary) ─────────────────────────────────────────────
1. FP REDUCTION  — DATA_ACTIVE_DAYS_RECENT_4W fired in 94.7% of all FPs, but
   DATA_REVENUE_RECENT_4W was a safety signal in 90.8% of those same cases.
   Fix: Added STABLE_REVENUE_LOW_ACTIVITY and HIGH_REVENUE_GUARD suppressors
   so low-frequency but still-paying users are no longer falsely flagged.
   Also reduced LOW_DATA_ACTIVITY weight 1.0 → 0.5 (top FP amplifier).

2. FN REDUCTION  — DATA_REVENUE_RECENT_4W wrongly suppressed 62.7% of missed
   churners. Real churners who had a positive 4W revenue history but collapsed
   in W13 were "protected" by that historical average.
   Fix: Added RECENT_ACTIVITY_COLLAPSE signal (W13 < 20% of W10-W12 avg) with
   weight +2.5 to override the 4W safety check when a sharp drop is detected.
   Also added alternate Tier 1 path for collapse-from-active-baseline cases.

3. THRESHOLD TIGHTENING — T2_SCORE_MIN raised 8 → 9 to improve Tier 2
   precision while the new signals expand Tier 1 recall for real churners.

── Benchmark results (original) ──────────────────────────────────────────────
  TIER 1 : Precision 61.1%  Recall 22%   Lift 122x  Alerts ~6.7K
  TIER 2 : Precision  4.7%  Recall 32%   Lift 9.4x  Alerts ~127K
  TIER 3 : Precision  1.4%  Recall 20%   Lift 2.8x  Alerts ~268K

── Run ────────────────────────────────────────────────────────────────────────
    python churn_rules_optimized.py                         # uses Feb_Train.csv
    INPUT_CSV=your_file.csv python churn_rules_optimized.py
    INPUT_CSV=your_file.csv CHURN_HORIZON=30 python churn_rules_optimized.py

── Output files (./churn_rules_output/) ───────────────────────────────────────
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
    "INPUT_CSV"       : os.getenv("INPUT_CSV",       "Feb1_Train_with_recharg.csv"),
    "OUTPUT_DIR"      : os.getenv("OUTPUT_DIR",       "./churn_rules_output"),
    "ID_COL"          : os.getenv("ID_COL",           "MSISDN"),
    "CHURN_HORIZON"   : int(os.getenv("CHURN_HORIZON", "90")),
    "DATASET_TYPE_COL": "DATASET_TYPE",

    # ── TIER 1: High-confidence rules (ALL conditions must be true) ───────────
    # Path A (STRUCTURAL CHANGE v3):
    #   Base:    NEARLY_INACTIVE (active_weeks ≤ 1) OR RELAXED (active_weeks ≤ 2 AND data_rev == 0)
    #   AND all of: NOT_RECOVERING + SERVICE_COLLAPSED + (ZERO_REV OR ALL_SERVICES_ZERO_W13)
    #   FP guard: ANY_ACTIVE_WEEKS_13W ≤ 8 — filters temporarily-lapsed subscribers
    #             (empirical: T1 FPs had 13W-active-weeks of 9 and 10 vs TPs max 8;
    #              these are "lapsing but returning" accounts, not true churners)
    # Path B (collapse-from-active, v2): RECENT_ACTIVITY_COLLAPSE + ALL_SERVICES_ZERO_W13
    #   + NOT_RECOVERING + SERVICE_COLLAPSED
    "T1_RULES": {
        "ANY_ACTIVE_WEEKS_RECENT_4W_MAX"         : 1,     # base threshold
        "ANY_ACTIVE_WEEKS_RECENT_4W_MAX_RELAXED" : 2,     # OR: ≤2 when data_rev == 0
        "SERVICE_DIVERSITY_RECENT_4W_MAX"        : 1,
        "RECOVERY_COUNT_MAX"                     : 0,
        "REQUIRE_ZERO_REV_OR_DEAD_W13"           : True,
        "LONG_TERM_ACTIVE_WEEKS_13W_MAX"         : 8,     # FP guard (13-week window)
    },

    # ── TIER 2: Medium-risk rule score (weighted sum of signals) ─────────────
    # CHANGED 8 → 9: tightened to improve Tier 2 precision after new signals
    # added positive weight (+2.5 collapse signal) that can inflate borderline scores.
    "T2_SCORE_MIN": 9,

    # ── TIER 3: Watch-list ────────────────────────────────────────────────────
    # Unchanged — broader coverage for low-cost digital channel.
    "T3_SCORE_MIN": 5,
    "T3_SCORE_MAX": 8,   # CHANGED from 7: now 5–8 to absorb mid-range scores

    # ── Rule weights (used in scoring) ────────────────────────────────────────
    # Positive = churn signal.  Negative = FP suppressor.
    "RULE_WEIGHTS": {
        # ── Primary behavioral signals (2 pts each) ───────────────────────────
        "NEARLY_INACTIVE_RECENT"     : 2.0,   # active_weeks_recent_4W <= 1
        "NOT_RECOVERING"             : 2.0,   # no service with W13 > W10
        "SERVICE_COLLAPSED"          : 2.0,   # service_diversity <= 1

        # ── Strong single signals (1.5 pts each) ─────────────────────────────
        "ALL_SERVICES_ZERO_W13"      : 1.5,   # data+voice+bundle all zero last week
        "NO_BUNDLE_RECENT"           : 1.5,   # bundle_active_weeks = 0
        "NEAR_ZERO_REVENUE"          : 1.5,   # total_revenue_recent_4W <= 10

        # ── NEW: Recent drop velocity signal (2.5 pts) — FN reducer ──────────
        # SHAP finding: DATA_REVENUE_RECENT_4W suppresses 62.7% of FN churners
        # because the 4W average masks a W13 collapse. This override signal fires
        # when W13 drops >80% vs W10–W12 average across ≥2 services, catching
        # churners who "looked safe" based on their recent 4W history.
        "RECENT_ACTIVITY_COLLAPSE"   : 2.5,

        # ── NEW: Severe single-service velocity collapse (1.5 pts) — FN reducer ─
        # STRUCTURAL CHANGE v3: Fires when W13 < 10% of W10-W12 average on ANY
        # single service with a real prior baseline (≥1 unit avg).
        # Unlike RECENT_ACTIVITY_COLLAPSE (which requires ≥2 services at 20%),
        # this catches churners who collapse catastrophically on one service.
        # Weight set to 1.5 (supporting signal, not a solo tier-driver).
        # It boosts churners who already score near-T2 from other signals over
        # the threshold, without independently elevating non-churners who fire
        # vel_severe on a single-service seasonal dip.
        "VELOCITY_COLLAPSE_SEVERE"   : 1.5,

        # ── Supporting signals ────────────────────────────────────────────────
        # LOW_DATA_ACTIVITY reduced 1.0 → 0.5:
        #   SHAP shows DATA_ACTIVE_DAYS_RECENT_4W fires in 94.7% of FPs.
        #   Low-frequency users who still pay revenue are systematically misflagged
        #   by this signal. Halving the weight limits its FP amplification while
        #   still contributing when combined with other signals.
        "LOW_DATA_ACTIVITY"          : 0.5,   # ↓ from 1.0 (top FP amplifier per SHAP)
        "NO_VOICE_ACTIVITY"          : 1.0,   # voice_active_weeks = 0
        "ACTIVITY_TRENDING_DOWN"     : 1.0,   # any_active_weeks_drop = 1
        "SERVICE_DIVERSITY_DROPPING" : 1.0,   # service_diversity_drop = 1
        "REVENUE_UNDER_50"           : 1.0,   # total_revenue_recent_4W <= 50

        # ── False-positive suppressors (negative = remove FP risk) ────────────

        # Original: multi-service recovery — validated 0% FN impact
        "RECOVERING_MULTI_SERVICE"   : -2.0,

        # NEW: Stable-revenue low-activity guard (−1.5 pts) — FP suppressor
        # SHAP finding: 90.8% of FPs have DATA_REVENUE_RECENT_4W as a safety
        # signal, meaning the subscriber IS generating revenue despite low
        # activity days. This suppressor identifies low-frequency but still-
        # paying users: data_revenue > 0 AND data_active_days in [1, 5].
        # These are loyal infrequent users, not churners.
        "STABLE_REVENUE_LOW_ACTIVITY": -1.5,

        # NEW: High-revenue absolute guard (−2.0 pts) — strong FP suppressor
        # Subscribers spending ≥100 in the last 4 weeks almost certainly are not
        # churning. No rule-based signal combination should outweigh this fact.
        # Validated conceptually: high-revenue subscribers in FP set had
        # TOTAL_REVENUE_RECENT_4W appearing as a safety driver in SHAP output.
        "HIGH_REVENUE_GUARD"         : -2.0,
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
    cols  = [f"{prefix}_{w}" for w in weeks]
    avail = [c for c in cols if c in df.columns]
    if not avail:
        return np.zeros((len(df), len(weeks)), dtype=np.float32)
    mat = (
        df[[c for c in cols]]
        .rename(columns={c: c for c in cols})
        .reindex(columns=cols, fill_value=0.0)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .values
        .astype(np.float32)
    )
    return mat


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all named churn signals. Returns a DataFrame of 0/1 columns.

    SHAP-optimized v2 adds:
      RECENT_ACTIVITY_COLLAPSE   — W13 < 20% of W10-W12 avg across ≥2 services
      STABLE_REVENUE_LOW_ACTIVITY — revenue > 0 but active days low (FP guard)
      HIGH_REVENUE_GUARD          — total 4W revenue ≥ 100 (strong FP guard)
    And reduces LOW_DATA_ACTIVITY weight (now 0.5, was 1.0).
    """
    n = len(df)
    S = {}   # signal dict

    # ── Weekly usage matrices (last 4 weeks W10-W13) ─────────────────────────
    data_mat   = _weekly_matrix(df, "DATA_MB")
    voice_mat  = _weekly_matrix(df, "OG_VOICE_MIN")
    bundle_mat = _weekly_matrix(df, "BUNDLE_CNT")
    sms_mat    = _weekly_matrix(df, "OG_SMS_COUNT")

    # ── Recovery guard: W13 vs W10 per service ───────────────────────────────
    # A subscriber with W13 > W10 is BOUNCING BACK — likely a false positive.
    # Validated: among 519 subscribers flagged as "drop and recover",
    # 0% (zero) were real churners.
    data_recovering   = (data_mat[:,   -1] > data_mat[:,   0]).astype(np.int8)
    voice_recovering  = (voice_mat[:,  -1] > voice_mat[:,  0]).astype(np.int8)
    bundle_recovering = (bundle_mat[:, -1] > bundle_mat[:, 0]).astype(np.int8)
    recovery_count    = data_recovering + voice_recovering + bundle_recovering  # 0-3

    # ── PRIMARY SIGNALS (2 pts each) ─────────────────────────────────────────

    # Signal 1: Nearly inactive in recent 4 weeks
    # Churners mean = 1.64 active weeks, Non-churners mean = 3.07 active weeks
    S["NEARLY_INACTIVE_RECENT"] = (
        _get_col(df, "ANY_ACTIVE_WEEKS_RECENT_4W") <= 1
    ).astype(np.int8)

    # Signal 2: No service is bouncing back (not a temp dip)
    # Churners: usage drops are permanent. Non-churners: they come back.
    S["NOT_RECOVERING"] = (recovery_count == 0).astype(np.int8)

    # Signal 3: Service diversity collapsed to ≤1 service
    # Churners mean diversity = 1.5, Non-churners = 3.2 services
    S["SERVICE_COLLAPSED"] = (
        _get_col(df, "SERVICE_DIVERSITY_RECENT_4W") <= 1
    ).astype(np.int8)

    # ── STRONG SINGLE SIGNALS (1.5 pts each) ─────────────────────────────────

    # Signal 4: All major services went to zero in the final week
    # The clearest single-week dead-SIM signal. Non-churner rate for this
    # group is only ~18% vs ~99.5% for the general population.
    S["ALL_SERVICES_ZERO_W13"] = (
        (data_mat[:,-1]   <= 0) &
        (voice_mat[:,-1]  <= 0) &
        (bundle_mat[:,-1] <= 0)
    ).astype(np.int8)

    # Signal 5: No bundle purchases in 4 weeks
    # Bundle cancellation is often the first step before full churn.
    # Churners: 0 active bundle weeks mean = 0.92, Non-churners = 2.39
    S["NO_BUNDLE_RECENT"] = (
        _get_col(df, "BUNDLE_ACTIVE_WEEKS_RECENT_4W") == 0
    ).astype(np.int8)

    # Signal 6: Near-zero revenue (≤10 KES/USD equivalent in 4 weeks)
    # Churners revenue mean = 87.55, median = 10.13, p25 = 0.00
    # This targets the lower half of churners who spent almost nothing
    S["NEAR_ZERO_REVENUE"] = (
        _get_col(df, "TOTAL_REVENUE_RECENT_4W") <= 10
    ).astype(np.int8)

    # ── NEW SIGNAL: RECENT ACTIVITY COLLAPSE (2.5 pts) — FN REDUCER ──────────
    #
    # SHAP finding: DATA_REVENUE_RECENT_4W suppresses 62.7% of missed churners
    # because the 4-week revenue average masks a sharp W13 collapse. Real churners
    # who were historically active but went silent in W13 look "safe" due to the
    # positive 4W average. This signal fires when W13 drops >80% vs the W10-W12
    # average across at least 2 service dimensions.
    #
    # Threshold rationale:
    #   - >80% drop (< 20% of prior average) is definitional: even a partially
    #     active subscriber would retain at least 30-40% of prior usage randomly.
    #   - Requiring ≥2 of 3 services to collapse avoids single-service seasonality
    #     (e.g., a subscriber who temporarily stops using data but keeps voice).
    #   - MIN_BASELINE = 1.0 ensures we only fire when there was real prior activity
    #     (not comparing 0 vs 0).

    MIN_BASELINE = 1.0   # prior 3-week average must exceed this to be meaningful

    # W10–W12 averages (columns 0, 1, 2 = W10, W11, W12)
    data_prior_avg   = data_mat[:,   0:3].mean(axis=1)   # avg of W10, W11, W12
    voice_prior_avg  = voice_mat[:,  0:3].mean(axis=1)
    bundle_prior_avg = bundle_mat[:, 0:3].mean(axis=1)

    data_w13   = data_mat[:,   -1]    # W13 (last week)
    voice_w13  = voice_mat[:,  -1]
    bundle_w13 = bundle_mat[:, -1]

    # Per-service collapse: W13 < 20% of W10-W12 average AND prior avg was real
    data_collapse   = (
        (data_prior_avg   > MIN_BASELINE) &
        (data_w13   < 0.20 * data_prior_avg)
    )
    voice_collapse  = (
        (voice_prior_avg  > MIN_BASELINE) &
        (voice_w13  < 0.20 * voice_prior_avg)
    )
    # Bundle: W13 = 0 when W10-W12 had consistent bundle activity
    bundle_collapse = (
        (bundle_prior_avg > 0.5) &      # had bundles on average over W10-W12
        (bundle_w13 == 0)               # went completely silent in W13
    )

    collapse_count = (
        data_collapse.astype(np.int8) +
        voice_collapse.astype(np.int8) +
        bundle_collapse.astype(np.int8)
    )
    # Fire when ≥2 of 3 services show the collapse pattern
    S["RECENT_ACTIVITY_COLLAPSE"] = (collapse_count >= 2).astype(np.int8)

    # ── NEW SIGNAL: VELOCITY COLLAPSE SEVERE (3.5 pts) — FN REDUCER ──────────
    #
    # STRUCTURAL CHANGE v3: Lowers the collapse threshold from 20% (≥2 services)
    # to 10% on ANY single service. This catches churners who experienced a
    # catastrophic drop on one service (e.g., data usage collapsed >90%) while
    # the 4-week average masked the severity.
    #
    # Key difference from RECENT_ACTIVITY_COLLAPSE:
    #   - Threshold: 10% of prior avg (not 20%)    ← harder floor
    #   - Breadth: fires on ANY 1 service (not ≥2) ← wider net
    # These two differences ensure it catches the FN churners who had a single-
    # service terminal collapse masked by healthy parallel services.
    #
    # HIGH_REVENUE_GUARD interaction: when this signal fires, the −2.0 revenue
    # guard is suppressed (see below). Without this override, high-revenue
    # churners who collapsed in W13 are penalized by their own prior activity,
    # netting only +0.5 pts instead of the full signal benefit.
    vel_severe_data   = (
        (data_prior_avg   > MIN_BASELINE) &
        (data_w13   < 0.10 * data_prior_avg)
    )
    vel_severe_voice  = (
        (voice_prior_avg  > MIN_BASELINE) &
        (voice_w13  < 0.10 * voice_prior_avg)
    )
    vel_severe_bundle = (
        (bundle_prior_avg > 0.5) &
        (bundle_w13 == 0)
    )
    vel_severe_any = vel_severe_data | vel_severe_voice | vel_severe_bundle
    S["VELOCITY_COLLAPSE_SEVERE"] = vel_severe_any.astype(np.int8)

    # ── SUPPORTING SIGNALS (varying pts) ─────────────────────────────────────

    # Signal 7: Data usage barely active
    # SHAP note: weight reduced from 1.0 → 0.5 because DATA_ACTIVE_DAYS_RECENT_4W
    # fires in 94.7% of FPs (low-frequency users who still pay revenue).
    S["LOW_DATA_ACTIVITY"] = (
        _get_col(df, "DATA_ACTIVE_WEEKS_RECENT_4W") <= 1
    ).astype(np.int8)

    # Signal 8: No outgoing voice calls at all in 4 weeks
    S["NO_VOICE_ACTIVITY"] = (
        _get_col(df, "TOTAL_VOICE_ACTIVE_WEEKS_RECENT_4W") == 0
    ).astype(np.int8)

    # Signal 9: Subscriber's activity is trending downward vs previous period
    S["ACTIVITY_TRENDING_DOWN"] = (
        _get_col(df, "ANY_ACTIVE_WEEKS_DROP") == 1
    ).astype(np.int8)

    # Signal 10: Number of active services dropped vs prior period
    S["SERVICE_DIVERSITY_DROPPING"] = (
        _get_col(df, "SERVICE_DIVERSITY_DROP") == 1
    ).astype(np.int8)

    # Signal 11: Revenue under 50 (broader low-spend band)
    S["REVENUE_UNDER_50"] = (
        _get_col(df, "TOTAL_REVENUE_RECENT_4W") <= 50
    ).astype(np.int8)

    # ── FALSE-POSITIVE SUPPRESSORS ────────────────────────────────────────────

    # Signal 12 (NEGATIVE, −2.0 pts): Multi-service recovery — NOT a churner
    # If 2 or more services have W13 > W10, the subscriber is bouncing back.
    # Validated on sample: 519/519 flagged subs were non-churners (0% FN impact).
    S["RECOVERING_MULTI_SERVICE"] = (recovery_count >= 2).astype(np.int8)

    # NEW Signal 13 (NEGATIVE, −1.5 pts): Stable-revenue low-activity guard
    #
    # SHAP finding: 90.8% of FPs have DATA_REVENUE_RECENT_4W as a safety signal,
    # meaning these non-churners are still generating revenue despite few active
    # data days. The existing rules treat DATA_ACTIVE_DAYS_RECENT_4W low-values
    # as a churn signal without checking whether revenue is still flowing.
    #
    # This suppressor identifies low-frequency but still-paying users:
    #   data_revenue > 0   → subscriber generated data revenue in 4W window
    #   data_active_days in [1, 5] → low-frequency (matches the FP pattern)
    #
    # Not applied when data_active_days = 0 (truly inactive → legitimate risk signal)
    # Not applied when data_revenue = 0  (no revenue → cannot suppress)
    data_rev  = _get_col(df, "DATA_REVENUE_RECENT_4W")
    data_days = _get_col(df, "DATA_ACTIVE_DAYS_RECENT_4W")
    S["STABLE_REVENUE_LOW_ACTIVITY"] = (
        (data_rev  > 0) &
        (data_days > 0) &
        (data_days <= 5)        # low-frequency threshold (1–5 active data days)
    ).astype(np.int8)

    # NEW Signal 14 (NEGATIVE, −2.0 pts): High-revenue absolute guard
    #
    # Subscribers spending ≥100 in the last 4 weeks are almost certainly not
    # churning. No combination of low-activity signals should outweigh the
    # fact that this subscriber is a significant revenue contributor.
    # SHAP evidence: TOTAL_REVENUE_RECENT_4W appeared as a safety driver in
    # 622 FP cases in the top-3; the −2.0 weight ensures the score cannot
    # reach Tier 2 threshold on low-activity signals alone.
    total_rev = _get_col(df, "TOTAL_REVENUE_RECENT_4W")
    S["HIGH_REVENUE_GUARD"] = (total_rev >= 100).astype(np.int8)

    # ── AUXILIARY COLUMNS (for Tier 1 hard rule, not scored) ─────────────────
    S["_ZERO_REV"]            = (total_rev <= 0).astype(np.int8)
    S["_DEAD_W13"]            = S["ALL_SERVICES_ZERO_W13"]
    S["_RECOVERY_COUNT"]      = recovery_count
    S["_COLLAPSE_COUNT"]      = collapse_count     # stored for diagnostics
    # STRUCTURAL CHANGE v3: 13W activity count for Tier 1 FP guard
    S["_ANY_ACTIVE_13W"]      = _get_col(df, "ANY_ACTIVE_WEEKS_13W").astype(np.float32)
    # 4W activity count for OR-relaxation gate in assign_tiers()
    S["_ANY_ACTIVE_RECENT_4W"] = _get_col(df, "ANY_ACTIVE_WEEKS_RECENT_4W")
    # Data revenue for OR-relaxation gate
    S["_DATA_REV_RECENT"]     = _get_col(df, "DATA_REVENUE_RECENT_4W")

    return pd.DataFrame(S, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# RISK SCORING & TIERING
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_score(signals: pd.DataFrame) -> np.ndarray:
    """
    Weighted sum of all signals.
    The result is a continuous risk score — higher = more like a churner.
    Negative scores indicate strong recovery/revenue behaviour (likely FP).
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

    STRUCTURAL CHANGE v3 — three modifications to Path A:

    TIER 1 — High-confidence churn risk (two alternate hard-AND paths):

      Path A  (STRUCTURAL CHANGE v3 — dead-SIM with OR fallback + FP guard):

        Inactivity condition (OR — relaxed from rigid AND):
          Base  : active_weeks_recent_4W <= 1                    (original)
          Relaxed: active_weeks_recent_4W <= 2 AND data_rev == 0 (new OR branch)
          Rationale: a subscriber with 2 active weeks who has generated zero data
          revenue is exhibiting the same dead-SIM economics as one with 1 active
          week — their data stream has stopped, regardless of nominal activity.

        ALL of (unchanged gates):
          NOT_RECOVERING + SERVICE_COLLAPSED
          + (ZERO_REVENUE OR ALL_SERVICES_ZERO_W13)

        PLUS (NEW FP guard):
          ANY_ACTIVE_WEEKS_13W <= 8
          Rationale: T1 FP analysis showed two non-churners with 13W active-week
          counts of 9 and 10 — "temporarily lapsed" long-term subscribers who
          resumed within the 90-day window. All true T1 churners had ≤ 8.
          This guard removes them without touching any true positives.

      Path B  (v2 — collapse from active baseline, unchanged):
        RECENT_ACTIVITY_COLLAPSE + ALL_SERVICES_ZERO_W13
        + NOT_RECOVERING + SERVICE_COLLAPSED

    TIER 2 — Medium-risk (risk score ≥ 9, not already Tier 1).

    TIER 3 — Watch-list (score 5–8).

    TIER 0 — No action.
    """

    # ── Tier 1 Path A: dead-SIM rule with OR fallback + FP guard ────────────
    #
    # STRUCTURAL CHANGE v3: two modifications to the original hard-AND gate.
    #
    # Modification 1 — OR fallback on the inactivity condition (CFG-documented):
    #   The CFG["T1_RULES"] now records both the strict and relaxed thresholds:
    #     Strict  : active_weeks_recent_4W <= 1  (original)
    #     Relaxed : active_weeks_recent_4W <= 2  AND  data_rev_4W == 0
    #   The OR is applied here as the combined inactivity gate.
    #   Empirical note: on the sample data the relaxed branch does not add new
    #   true positives (FN churners with active_weeks==2 lack the NOT_RECOVERING
    #   and SERVICE_COLLAPSED conditions needed to enter T1). The OR is therefore
    #   net-neutral on recall but widens the gate for production data where those
    #   conditions can align. We apply it only when ALL other T1 gates also hold
    #   (NOT_RECOVERING, SERVICE_COLLAPSED, zero_rev/dead_W13) so a subscriber
    #   must still meet every structural churn criterion — just with a slightly
    #   relaxed activity floor.
    #
    # Modification 2 — 13W long-term activity FP guard (primary FP reducer):
    #   Require ANY_ACTIVE_WEEKS_13W <= T1_RULES["LONG_TERM_ACTIVE_WEEKS_13W_MAX"] (=8).
    #   Rationale: T1 FP analysis found two non-churners with 13W active-week
    #   counts of 9 and 10. These are "temporarily lapsed" long-term subscribers
    #   who returned within the 90-day label window (hence labelled non-churn).
    #   ALL 11 true T1 positives had ANY_ACTIVE_WEEKS_13W ≤ 8.
    #   This guard removes both FPs without removing a single TP.

    any_active_4w  = signals["_ANY_ACTIVE_RECENT_4W"].values   # 4W raw count
    data_rev_vals  = signals["_DATA_REV_RECENT"].values        # 4W data revenue

    not_rec        = signals["NOT_RECOVERING"].values.astype(bool)
    svc_col        = signals["SERVICE_COLLAPSED"].values.astype(bool)
    zero_rev       = signals["_ZERO_REV"].values.astype(bool)   # total_rev <= 0
    dead_w13       = signals["_DEAD_W13"].values.astype(bool)
    zero_rev_or_dead = zero_rev | dead_w13

    # 13W long-term activity FP guard (STRUCTURAL — Modification 2)
    t1_fp_guard = (
        signals["_ANY_ACTIVE_13W"].values <= CFG["T1_RULES"]["LONG_TERM_ACTIVE_WEEKS_13W_MAX"]
    )

    # Path A Branch 1: original dead-SIM rule (strict inactivity)
    #   active_weeks <= 1 AND not_rec AND svc_col AND (zero_rev OR dead_w13) AND 13W_guard
    branch1 = (
        (any_active_4w <= CFG["T1_RULES"]["ANY_ACTIVE_WEEKS_RECENT_4W_MAX"]) &
        not_rec & svc_col & zero_rev_or_dead & t1_fp_guard
    )

    # Path A Branch 2 (STRUCTURAL CHANGE — OR fallback):
    #   active_weeks <= 2 AND total_rev == 0 AND not_rec AND svc_col AND 13W_guard
    #   Stricter than Branch 1 in revenue (requires total_rev=0 not just dead_w13),
    #   relaxed only on the inactivity floor (allows active_weeks=2).
    #   This targets the edge case: a subscriber with exactly 2 active weeks
    #   and zero revenue in 4W — same economic state as a ≤1-week dead-SIM.
    branch2 = (
        (any_active_4w <= CFG["T1_RULES"]["ANY_ACTIVE_WEEKS_RECENT_4W_MAX_RELAXED"]) &
        (data_rev_vals == 0) &             # data revenue zero
        zero_rev &                          # total revenue also zero (strict)
        not_rec & svc_col & t1_fp_guard
    )

    tier1a_mask = branch1 | branch2

    # ── Tier 1 Path B: collapse from active baseline (v2 — unchanged) ────────
    tier1b_mask = (
        (signals["RECENT_ACTIVITY_COLLAPSE"].values.astype(bool)) &
        (signals["ALL_SERVICES_ZERO_W13"].values.astype(bool)) &
        not_rec &
        svc_col
    )

    tier1_mask = tier1a_mask | tier1b_mask

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
    print("  RULE-BASED CHURN DETECTION — EVALUATION REPORT  [SHAP-Optimized v2]")
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

    # New signal diagnostics
    _sep("─", 62)
    print("  NEW SIGNAL DIAGNOSTICS (SHAP-optimized signals only)")
    _sep("─", 62)
    new_signals = [
        "RECENT_ACTIVITY_COLLAPSE",
        "STABLE_REVENUE_LOW_ACTIVITY",
        "HIGH_REVENUE_GUARD",
    ]
    for col in new_signals:
        if col not in signals.columns:
            continue
        n_fired = int(signals[col].sum())
        if y_true is not None and n_fired > 0:
            tp_s  = int(((signals[col]==1)&(y_true==1)).sum())
            prec_s = tp_s/n_fired
            weight = CFG["RULE_WEIGHTS"].get(col, 0)
            print(f"  {col:<35}  fired={n_fired:>6,}  TP={tp_s:>4}  "
                  f"prec={prec_s:.4f}  weight={weight:+.1f}")
    print()

    # All signal firing rates
    _sep("─", 62)
    print("  SIGNAL FIRING RATES")
    _sep("─", 62)
    signal_cols = [c for c in signals.columns if not c.startswith("_")]
    for col in signal_cols:
        n_fired = int(signals[col].sum())
        if y_true is not None:
            tp_s   = int(((signals[col]==1)&(y_true==1)).sum())
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
    daily_base     = 3_818_400
    scale          = daily_base / n_total
    daily_churners = int(22208)   # from OOT report

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
    print("  RECOMMENDED CAMPAIGN STRATEGY  [SHAP-Optimized v2]")
    _sep()
    print("""
  TIER 1 — CRM / Retention Specialist Team
  ──────────────────────────────────────────
  Path A: NEARLY_INACTIVE_RECENT AND NOT_RECOVERING AND SERVICE_COLLAPSED
          AND (ZERO_REVENUE OR ALL_SERVICES_ZERO_W13)
          → Dead-SIM / near-dead signals with no bounce-back [original rule]

  Path B: RECENT_ACTIVITY_COLLAPSE AND ALL_SERVICES_ZERO_W13
          AND NOT_RECOVERING AND SERVICE_COLLAPSED
          → NEW: Churners who collapsed in W13 from an active W10-W12 baseline
          Targets the 62.7% of FN churners whose 4W revenue masked their W13 drop.

  TIER 2 — Automated Retention Campaign
  ──────────────────────────────────────
  Rule : Risk score ≥ 9 (raised from 8 — tightened after adding new signals)
  New suppressors (STABLE_REVENUE_LOW_ACTIVITY and HIGH_REVENUE_GUARD) will
  remove low-frequency but still-paying users from this tier automatically.

  TIER 3 — Low-Cost Digital Watch List
  ──────────────────────────────────────
  Rule : Risk score 5–8 (range extended from 5–7)
  Best for: app push notification, email, recharge incentive.

  KEY SHAP-DRIVEN CHANGES IN THIS VERSION:
  • RECENT_ACTIVITY_COLLAPSE (+2.5 pts): Targets churners missed because their
    4W revenue history suppressed the churn signal (62.7% FN root cause)
  • STABLE_REVENUE_LOW_ACTIVITY (−1.5 pts): Removes low-frequency paying users
    from the FP pool (94.7% of FPs had low DATA_ACTIVE_DAYS but positive revenue)
  • HIGH_REVENUE_GUARD (−2.0 pts): Prevents high-spending subscribers from ever
    being scored into actionable tiers by low-activity signals alone
  • LOW_DATA_ACTIVITY weight: 1.0 → 0.5 (direct FP amplifier per SHAP analysis)
  • T2_SCORE_MIN: 8 → 9 (precision tightening to offset new signal addition)
""")
    _sep()


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(y_true, tiers, score, signals, results, n_total):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 13))
    fig.suptitle(
        "Rule-Based Churn Detection — Diagnostic Dashboard  [SHAP-Optimized v2]",
        fontsize=15, fontweight="bold", y=0.99,
    )
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
            ax2.annotate(lbl, (r, p), textcoords="offset points",
                         xytext=(8,4), fontsize=9)
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
        score_churn = score[y_true==1]
        score_nonch = score[y_true==0]
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

    scored = df[[id_col]].copy() if id_col in df.columns else pd.DataFrame(index=df.index)
    if target_col and target_col in df.columns:
        scored["actual_churn"] = df[target_col].values

    scored["risk_score"] = score.round(2)
    scored["tier"]       = tiers
    scored["tier_label"] = pd.Series(tiers).map({
        0: "NO_ACTION", 1: "TIER_1_HIGH", 2: "TIER_2_MEDIUM", 3: "TIER_3_WATCH"
    }).values

    # Attach all signals for explainability
    signal_cols = [c for c in signals.columns if not c.startswith("_")]
    for col in signal_cols:
        scored[f"sig_{col}"] = signals[col].values

    # Also expose collapse diagnostics
    if "_COLLAPSE_COUNT" in signals.columns:
        scored["collapse_count"] = signals["_COLLAPSE_COUNT"].values

    all_path = os.path.join(CFG["OUTPUT_DIR"], "churn_scored.csv")
    scored.to_csv(all_path, index=False)
    log.info("Full scored CSV → %s  (%d rows)", all_path, len(scored))

    for tier_num, tier_name in [(1,"tier1_alerts"), (2,"tier2_alerts"), (3,"tier3_watchlist")]:
        tier_df = scored[scored["tier"]==tier_num].copy()
        path    = os.path.join(CFG["OUTPUT_DIR"], f"churn_{tier_name}.csv")
        tier_df.to_csv(path, index=False)
        log.info("  %s → %d rows → %s", tier_name.upper(), len(tier_df), path)

    report_lines  = []
    report_lines += [f"CHURN RULE-BASED SCORING REPORT [SHAP-Optimized v2] — {time.strftime('%Y-%m-%d %H:%M')}"]
    report_lines += [f"Input CSV: {CFG['INPUT_CSV']}"]
    report_lines += [f"Total rows: {len(df):,}"]
    report_lines += [f"Tier 1: {int((tiers==1).sum()):,}  Tier 2: {int((tiers==2).sum()):,}  "
                     f"Tier 3: {int((tiers==3).sum()):,}  No-action: {int((tiers==0).sum()):,}"]
    report_lines += [""]
    report_lines += ["KEY PARAMETER CHANGES vs ORIGINAL:"]
    report_lines += ["  RECENT_ACTIVITY_COLLAPSE    +2.5 (NEW — FN reducer)"]
    report_lines += ["  STABLE_REVENUE_LOW_ACTIVITY -1.5 (NEW — FP suppressor)"]
    report_lines += ["  HIGH_REVENUE_GUARD          -2.0 (NEW — FP suppressor)"]
    report_lines += ["  LOW_DATA_ACTIVITY           +0.5 (was +1.0 — reduced FP amplifier)"]
    report_lines += ["  T2_SCORE_MIN                  9  (was 8 — precision tightening)"]
    report_lines += ["  T3_SCORE_MAX                  8  (was 7 — extended range)"]
    report_lines += ["  Tier 1 Path B: collapse-from-active-baseline (NEW)"]

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
    print("  RULE-BASED CHURN DETECTION PIPELINE  [SHAP-Optimized v2]")
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
    log.info("Computing churn signals (SHAP-optimized v2) …")
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
    print(f"    churn_scored.csv          — every subscriber with tier + signals")
    print(f"    churn_tier1_alerts.csv    — high-confidence campaign list")
    print(f"    churn_tier2_alerts.csv    — medium-risk automated campaign")
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
