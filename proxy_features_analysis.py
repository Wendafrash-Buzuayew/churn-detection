"""
proxy_features_analysis.py
===========================
Derives frustration & decay proxy features from the 13-week
transactional CSV (usage, revenue, recharge, tenure) and validates
each feature's churn-separation power against LABEL_CHURN_90D.

No external data required — all features built from columns already
present in Sample_data_full_feature CSV.

Run:
    python3 scripts/proxy_features_analysis.py
    python3 scripts/proxy_features_analysis.py --csv path/to/your_file.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CSV = Path(__file__).parent.parent / (
    "attached_assets/Sample_data_full_feature_(1)_1783009657718.csv"
)

parser = argparse.ArgumentParser(description="Churn proxy feature analysis")
parser.add_argument("--csv", default=str(DEFAULT_CSV),
                    help="Path to the feature CSV file")
args = parser.parse_args()

CSV_PATH = Path(args.csv)
if not CSV_PATH.exists():
    print(f"\n[ERROR] File not found: {CSV_PATH}")
    print("Usage: python3 scripts/proxy_features_analysis.py --csv /path/to/your_file.csv\n")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  Churn Proxy Feature Analysis  —  Frustration & Decay Signals")
print("=" * 70)
print(f"\n  Loading: {CSV_PATH.name}")

df = pd.read_csv(CSV_PATH)
TARGET = "LABEL_CHURN_90D"

print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
print(f"  Churners: {df[TARGET].sum():,}  ({df[TARGET].mean()*100:.2f}% base rate)")
print(f"  TRAIN: {(df['DATASET_TYPE']=='TRAIN').sum():,}  |  TEST: {(df['DATASET_TYPE']=='TEST').sum():,}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def col(name: str, default: float = 0.0) -> pd.Series:
    """Safe column accessor — returns zeros if column absent."""
    return df[name].fillna(0.0) if name in df.columns else pd.Series(default, index=df.index)

def effect_size(series: pd.Series, label: pd.Series) -> float:
    """Cohen's d: |mean(churn) - mean(non-churn)| / pooled_std"""
    c = series[label == 1]
    nc = series[label == 0]
    pooled_std = np.sqrt((c.std() ** 2 + nc.std() ** 2) / 2)
    return abs(c.mean() - nc.mean()) / (pooled_std + 1e-9)

def top_decile_lift(series: pd.Series, label: pd.Series) -> float:
    """Churn rate in top 10% of score / base churn rate."""
    threshold = series.quantile(0.90)
    top = label[series >= threshold]
    base = label.mean()
    return top.mean() / (base + 1e-9)

# ─────────────────────────────────────────────────────────────────────────────
# Available weekly columns (W10–W13 only in this CSV)
# ─────────────────────────────────────────────────────────────────────────────

WEEKS = ["W10", "W11", "W12", "W13"]
EPS = 1e-6

feat = pd.DataFrame(index=df.index)

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 1 — Multi-SIM / OG-IC Behavioural Drift
#   Logic: A subscriber migrating primary traffic to a competitor SIM
#   continues to receive calls (IC stays up) but stops making OG calls.
#   OG share drops while IC share is stable — invisible to aggregate metrics.
# ─────────────────────────────────────────────────────────────────────────────

og_recent = col("OG_VOICE_MIN_W13") + col("OG_VOICE_MIN_W12")          # last 2 weeks
og_prev   = col("OG_VOICE_MIN_W11") + col("OG_VOICE_MIN_W10")          # prior 2 weeks
ic_recent = col("IC_VOICE_MIN_W13") + col("IC_VOICE_MIN_W12")
ic_prev   = col("IC_VOICE_MIN_W11") + col("IC_VOICE_MIN_W10")

og_share_recent = og_recent / (og_recent + ic_recent + EPS)
og_share_prev   = og_prev   / (og_prev   + ic_prev   + EPS)

feat["F01_OG_SHARE_DRIFT"] = og_share_prev - og_share_recent
# Positive = OG share falling (churn signal). Matches your churn_advanced_features ES=1.86

feat["F02_VOICE_ASYMMETRY_RECENT"] = (
    (ic_recent - og_recent) / (ic_recent + og_recent + EPS)
)
# +1 = only receiving calls (ghost subscriber). Churners trend toward +0.5 to +1.0

feat["F03_IC_ALIVE_OG_DEAD_FLAG"] = (
    ((ic_recent > 0) & (og_recent == 0)).astype(int)
)
# Binary: still reachable but stopped making calls — SIM-alive-but-abandoned

# SMS version of OG share drift (SMS asymmetry often precedes voice by 2-4 weeks)
og_sms_r = col("OG_SMS_COUNT_W13") + col("OG_SMS_COUNT_W12")
og_sms_p = col("OG_SMS_COUNT_W11") + col("OG_SMS_COUNT_W10")
ic_sms_r = col("IC_SMS_COUNT_W13") + col("IC_SMS_COUNT_W12")
ic_sms_p = col("IC_SMS_COUNT_W11") + col("IC_SMS_COUNT_W10")

og_sms_share_r = og_sms_r / (og_sms_r + ic_sms_r + EPS)
og_sms_share_p = og_sms_p / (og_sms_p + ic_sms_p + EPS)
feat["F04_SMS_OG_SHARE_DRIFT"] = og_sms_share_p - og_sms_share_r

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 2 — Network Friction Proxies (from usage pattern disruption)
#   Logic: When a network degrades, data sessions become erratic — the
#   subscriber uses data less frequently but may still top up (non-committed
#   sessions). Activity days vs active weeks divergence flags this.
# ─────────────────────────────────────────────────────────────────────────────

# Days-per-active-week ratio: healthy = 5-7 days/week, degraded = 1-2 days
data_days_per_week = col("DATA_ACTIVE_DAYS_RECENT_4W") / (
    col("DATA_ACTIVE_WEEKS_RECENT_4W") + EPS
)
feat["F05_DATA_SESSION_DENSITY"] = data_days_per_week
# Low value = subscriber appears only sporadically → likely using competitor SIM

# Data usage collapse: W13 vs W10 drop magnitude
data_w10 = col("DATA_MB_W10")
data_w13 = col("DATA_MB_W13")
feat["F06_DATA_COLLAPSE_RATIO"] = 1.0 - (data_w13 / (data_w10 + EPS)).clip(0, 2)
# Near 1.0 = total collapse, negative = recovery (growing subscriber)

# Cross-service coherence: are voice and data declining TOGETHER?
# Correlated decline = organic churn. Divergent decline = service-specific issue
data_drop = col("DATA_MB_W12") - col("DATA_MB_W13")
voice_drop = col("OG_VOICE_MIN_W12") - col("OG_VOICE_MIN_W13")
# Normalize drops by their respective W12 baselines
data_drop_norm  = data_drop  / (col("DATA_MB_W12")       + EPS)
voice_drop_norm = voice_drop / (col("OG_VOICE_MIN_W12")  + EPS)
# Feature: both dropping simultaneously (product of normalized drops)
feat["F07_DATA_VOICE_JOINT_COLLAPSE"] = (
    data_drop_norm.clip(-1, 5) * voice_drop_norm.clip(-1, 5)
)
# High positive = both metrics dropped together in the last week

# Erratic data pattern: high CV across W10-W13 relative to mean
data_mat = np.column_stack([
    col("DATA_MB_W10"), col("DATA_MB_W11"),
    col("DATA_MB_W12"), col("DATA_MB_W13"),
])
data_cv = np.std(data_mat, axis=1) / (np.mean(data_mat, axis=1) + EPS)
feat["F08_DATA_CV_4W"] = np.clip(data_cv, 0, 10)
# High CV with low mean = erratic, low-commitment usage (frustration proxy)

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 3 — Service Quality Proxies (activity frequency disruption)
#   Logic: A subscriber who drops from 7 active days/week to 2
#   is experiencing friction. Zero-week streaks over W10-W13 flag this.
# ─────────────────────────────────────────────────────────────────────────────

# Count of zero-usage weeks across the 4-week window per service
zero_data  = (data_mat == 0).sum(axis=1)
voice_mat  = np.column_stack([
    col("OG_VOICE_MIN_W10"), col("OG_VOICE_MIN_W11"),
    col("OG_VOICE_MIN_W12"), col("OG_VOICE_MIN_W13"),
])
zero_voice = (voice_mat == 0).sum(axis=1)
sms_mat = np.column_stack([
    col("OG_SMS_COUNT_W10"), col("OG_SMS_COUNT_W11"),
    col("OG_SMS_COUNT_W12"), col("OG_SMS_COUNT_W13"),
])
zero_sms = (sms_mat == 0).sum(axis=1)

feat["F09_ZERO_SERVICE_WEEKS_4W"] = zero_data + zero_voice + zero_sms
# Max = 12 (all 3 services dead all 4 weeks). High value = hibernating subscriber

# Multi-service simultaneous zero in W13 (highest validated ES in your pipeline)
feat["F10_MULTI_SVC_ZERO_W13"] = (
    (col("DATA_MB_W13") <= 0).astype(int)
    + (col("OG_VOICE_MIN_W13") <= 0).astype(int)
    + (col("OG_SMS_COUNT_W13") <= 0).astype(int)
    + (col("BUNDLE_CNT_W13") <= 0).astype(int)
)
# 4 = completely dark subscriber in most recent week

# Service diversity drop (from pre-computed column)
feat["F11_SERVICE_DIVERSITY_DROP"] = col("SERVICE_DIVERSITY_DROP")

# Consecutive zero weeks (pre-computed, use max across services)
feat["F12_MAX_CONSEC_ZERO_ANY"] = np.maximum.reduce([
    col("DATA_CONSECUTIVE_ZERO_WEEKS_RECENT"),
    col("OG_VOICE_CONSECUTIVE_ZERO_WEEKS_RECENT"),
    col("OG_SMS_CONSECUTIVE_ZERO_WEEKS_RECENT"),
    col("BUNDLE_CONSECUTIVE_ZERO_WEEKS_RECENT"),
])

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 4 — Silent Complaint Proxies (recharge/wallet dynamics)
#   Logic: Frustrated customers move from predictable bundle purchases
#   to erratic micro-recharges before stopping entirely.
# ─────────────────────────────────────────────────────────────────────────────

# Bundle purchase volatility: CV of bundle counts W10-W13
bundle_mat = np.column_stack([
    col("BUNDLE_CNT_W10"), col("BUNDLE_CNT_W11"),
    col("BUNDLE_CNT_W12"), col("BUNDLE_CNT_W13"),
])
bundle_cv = np.std(bundle_mat, axis=1) / (np.mean(bundle_mat, axis=1) + EPS)
feat["F13_BUNDLE_PURCHASE_CV"] = np.clip(bundle_cv, 0, 10)
# High CV with declining mean = erratic micro-bundle behaviour

# Revenue degradation ratio: recent 4W revenue vs peak implied by W10
# Using TOTAL_REVENUE_RECENT_SHARE_13W if available, else derive from BUNDLE_REVENUE
rev_recent = col("TOTAL_REVENUE_RECENT_4W")
bundle_rev = col("BUNDLE_REVENUE_RECENT_4W")
voice_rev  = col("VOICE_REVENUE_RECENT_4W")
total_rev_proxy = rev_recent + bundle_rev + voice_rev

# Revenue per active week (collapses when subscriber stops buying)
active_weeks = col("DATA_ACTIVE_WEEKS_RECENT_4W").clip(1, 4)
feat["F14_REV_PER_ACTIVE_WEEK"] = np.log1p(total_rev_proxy / active_weeks)

# Bundle trend: is the subscriber buying fewer bundles over W10-W13?
bundle_w10 = col("BUNDLE_CNT_W10")
bundle_w13 = col("BUNDLE_CNT_W13")
feat["F15_BUNDLE_DECAY_RATIO"] = 1.0 - (bundle_w13 / (bundle_w10 + EPS)).clip(0, 2)

# Long-drop percentage on revenue (pre-computed — peak to trough over 13W)
feat["F16_REVENUE_LONG_DROP_PCT"] = col("TOTAL_REVENUE_LONG_DROP_PCT")

# W13 vs W12 revenue cliff: sudden single-week revenue stop
feat["F17_REVENUE_W13_CLIFF"] = (
    col("BUNDLE_CNT_W13_VS_W12_DROP_PCT").clip(-2, 2)
)

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 5 — Composite Distress Index
#   Combines the strongest individual signals into one actionable score.
# ─────────────────────────────────────────────────────────────────────────────

# Normalise key signals to [0,1] before combining
def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + EPS)

feat["F18_COMPOSITE_DISTRESS_IDX"] = (
    0.25 * minmax(feat["F01_OG_SHARE_DRIFT"].clip(0, 1))
    + 0.20 * minmax(feat["F10_MULTI_SVC_ZERO_W13"].astype(float))
    + 0.15 * minmax(feat["F12_MAX_CONSEC_ZERO_ANY"].clip(0, 4))
    + 0.15 * minmax(feat["F06_DATA_COLLAPSE_RATIO"].clip(0, 1))
    + 0.10 * minmax(feat["F08_DATA_CV_4W"].clip(0, 5))
    + 0.10 * minmax(feat["F04_SMS_OG_SHARE_DRIFT"].clip(0, 1))
    + 0.05 * minmax(feat["F03_IC_ALIVE_OG_DEAD_FLAG"].astype(float))
)

# ─────────────────────────────────────────────────────────────────────────────
# Attach label and run validation
# ─────────────────────────────────────────────────────────────────────────────

feat["MSISDN"]           = df["MSISDN"].values
feat["DATASET_TYPE"]     = df["DATASET_TYPE"].values
feat[TARGET]             = df[TARGET].values

label = df[TARGET]

FEATURE_COLS = [c for c in feat.columns if c.startswith("F")]

print("─" * 70)
print(f"  {'Feature':<35} {'Effect Size':>12} {'Top-10% Lift':>14} {'Churn Mean':>12} {'Non-Churn Mean':>15}")
print("─" * 70)

results = []
for fcol in FEATURE_COLS:
    series = feat[fcol].fillna(0.0).astype(float)
    es   = effect_size(series, label)
    lift = top_decile_lift(series, label)
    cm   = series[label == 1].mean()
    ncm  = series[label == 0].mean()
    results.append({"feature": fcol, "effect_size": es, "lift": lift,
                    "churn_mean": cm, "non_churn_mean": ncm})
    flag = " ★" if es >= 0.5 else ("  " if es >= 0.2 else "  ")
    print(f"  {fcol:<35} {es:>12.3f} {lift:>14.1f}×  {cm:>12.4f} {ncm:>15.4f}{flag}")

results_df = pd.DataFrame(results).sort_values("effect_size", ascending=False)

print("─" * 70)
print(f"\n  ★ = Cohen's d ≥ 0.50  (strong separation from your 0.5% base rate)\n")

# ─────────────────────────────────────────────────────────────────────────────
# Summary by churn label
# ─────────────────────────────────────────────────────────────────────────────

print("─" * 70)
print("  Top 5 features by Effect Size")
print("─" * 70)
for _, row in results_df.head(5).iterrows():
    print(f"  {row['feature']:<35}  ES={row['effect_size']:.3f}  Lift={row['lift']:.1f}×")

print("\n─── Composite Distress Index — breakdown by DATASET_TYPE ─────────")
for split in ["TRAIN", "TEST"]:
    sub = feat[feat["DATASET_TYPE"] == split]
    lbl = sub[TARGET]
    idx = sub["F18_COMPOSITE_DISTRESS_IDX"]
    c_mean  = idx[lbl == 1].mean()
    nc_mean = idx[lbl == 0].mean()
    es      = effect_size(idx, lbl)
    lift    = top_decile_lift(idx, lbl)
    print(f"  {split:5s}  Churn={c_mean:.4f}  NonChurn={nc_mean:.4f}  "
          f"ES={es:.3f}  Top-10%-Lift={lift:.1f}×")

print("\n─── Top-decile distress: what % are actual churners? ──────────────")
threshold_90 = feat["F18_COMPOSITE_DISTRESS_IDX"].quantile(0.90)
top_decile   = feat[feat["F18_COMPOSITE_DISTRESS_IDX"] >= threshold_90]
pct_churn    = top_decile[TARGET].mean() * 100
base_rate    = label.mean() * 100
print(f"  Threshold (90th pct): {threshold_90:.4f}")
print(f"  Churners in top-10%: {pct_churn:.2f}%  (base rate: {base_rate:.2f}%)")
print(f"  Lift: {pct_churn / base_rate:.1f}×")
print(f"  Subscribers flagged: {len(top_decile):,}")

# ─────────────────────────────────────────────────────────────────────────────
# Export enriched file
# ─────────────────────────────────────────────────────────────────────────────

output_path = Path("scripts/output_proxy_features.csv")
export_cols = ["MSISDN", "DATASET_TYPE", TARGET] + FEATURE_COLS
feat[export_cols].to_csv(output_path, index=False)
print(f"\n  Output saved → {output_path}")
print(f"  Columns: {len(export_cols)}  (MSISDN + label + {len(FEATURE_COLS)} proxy features)\n")
print("  To merge back into your main pipeline:")
print("  >>> enriched = pd.read_csv('scripts/output_proxy_features.csv')")
print("  >>> final_df = main_df.merge(enriched[['MSISDN'] + proxy_cols], on='MSISDN', how='left')\n")
print("=" * 70)
print("  Done.")
print("=" * 70)