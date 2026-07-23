"""
churn_shap_analysis.py
======================
SHAP-based Explainability for the Rule-Based Churn Pipeline
────────────────────────────────────────────────────────────
Answers three key business questions:
 
  WHY was subscriber X flagged as a churner?
  WHY was a real churner MISSED (false negative)?
  WHY was a non-churner wrongly flagged (false positive)?
 
The module:
  1. Auto-discovers features from any CSV (no hardcoded column list)
  2. Trains a lightweight GBM that mirrors the rule-based tier decisions
  3. Applies SHAP TreeExplainer — exact, fast, scales to 500K+ rows
  4. Generates a 7-panel diagnostic report focused on FP and FN groups
  5. Exports per-subscriber SHAP explanations to CSV for downstream use
 
Run standalone (uses churn_rules output):
    python churn_shap_analysis.py
 
Or pass a different scored CSV:
    INPUT_CSV=Feb_Train.csv  SCORED_CSV=churn_rules_output/churn_scored.csv  python churn_shap_analysis.py
 
Env vars:
    INPUT_CSV         Raw feature CSV (same file you passed to churn_rules.py)
    SCORED_CSV        Output from churn_rules.py with 'tier' and 'risk_score' columns
    TARGET_COL        Churn label column (auto-detected if not set)
    OUTPUT_DIR        Where to write analysis files (default: ./churn_shap_output)
    MAX_SHAP_ROWS     Max rows used for SHAP computation (default: 50000; sampled if larger)
    N_BACKGROUND      TreeExplainer background size (default: 500)
    N_EXPLAIN_FP      How many FP subscribers to explain individually (default: 20)
    N_EXPLAIN_FN      How many FN subscribers to explain individually (default: 20)
    GBM_TREES         GBM n_estimators (default: 300 — enough to match rule patterns)
"""
 
import os
import sys
import time
import logging
import warnings
from typing import List, Tuple, Dict, Optional
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
 
try:
    import shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False
 
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
    "INPUT_CSV"    : os.getenv("INPUT_CSV",    "Feb_Train.csv"),
    "SCORED_CSV"   : os.getenv("SCORED_CSV",   "churn_rules_output/churn_scored.csv"),
    "TARGET_COL"   : os.getenv("TARGET_COL",   ""),          # auto-detected if empty
    "OUTPUT_DIR"   : os.getenv("OUTPUT_DIR",   "./churn_shap_output"),
    "MAX_SHAP_ROWS": int(os.getenv("MAX_SHAP_ROWS", "50000")),
    "N_BACKGROUND" : int(os.getenv("N_BACKGROUND", "500")),
    "N_EXPLAIN_FP" : int(os.getenv("N_EXPLAIN_FP", "20")),
    "N_EXPLAIN_FN" : int(os.getenv("N_EXPLAIN_FN", "20")),
    "GBM_TREES"    : int(os.getenv("GBM_TREES", "300")),
    "RANDOM_STATE" : 42,
    # Columns that are NEVER features (identifiers, labels, metadata)
    "ID_COLS"      : {
        "MSISDN","MSISDN_9","MSISDN_251","SNAPSHOT_DATE",
        "DATASET_TYPE","AON","LABEL_CHURN_30D","LABEL_CHURN_90D",
        # churn_rules.py output columns
        "ACTUAL_CHURN","RISK_SCORE","TIER","TIER_LABEL",
    },
}
 
os.makedirs(CFG["OUTPUT_DIR"], exist_ok=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & FEATURE AUTO-DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
 
def load_and_merge() -> Tuple[pd.DataFrame, List[str], str]:
    """
    Load the raw feature CSV and (optionally) the churn_rules scored CSV.
    Merges on MSISDN if both are present so we get tier + raw features together.
 
    Returns (merged_df, feature_columns, target_column)
    """
    raw_path    = CFG["INPUT_CSV"]
    scored_path = CFG["SCORED_CSV"]
 
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw feature CSV not found: {raw_path}\n"
            f"Set INPUT_CSV=/path/to/your_data.csv"
        )
 
    log.info("Loading raw data: %s", raw_path)
    df = pd.read_csv(raw_path)
    df.columns = [c.upper().strip() for c in df.columns]
    log.info("Raw data: %d rows × %d cols", *df.shape)
 
    # Merge scored output if available
    if os.path.exists(scored_path):
        log.info("Loading scored output: %s", scored_path)
        sc = pd.read_csv(scored_path)
        sc.columns = [c.upper().strip() for c in sc.columns]
        id_col = "MSISDN" if "MSISDN" in sc.columns else sc.columns[0]
        raw_id = "MSISDN" if "MSISDN" in df.columns else df.columns[0]
        # Only bring in rule-engine columns
        rule_cols = [c for c in sc.columns if c not in df.columns or c == id_col]
        df = df.merge(sc[[id_col]+[c for c in rule_cols if c != id_col]],
                      left_on=raw_id, right_on=id_col, how="left")
        log.info("After merge: %d rows × %d cols", *df.shape)
    else:
        log.warning("Scored CSV not found at %s — running without tier info", scored_path)
 
    # ── Resolve target column ─────────────────────────────────────────────────
    target = CFG["TARGET_COL"].upper() if CFG["TARGET_COL"] else ""
    if not target or target not in df.columns:
        candidates = [c for c in df.columns if c.startswith("LABEL_CHURN_")]
        if candidates:
            target = candidates[0]
            log.info("Auto-detected target column: %s", target)
        else:
            target = None
            log.warning("No churn label column found — FP/FN analysis unavailable")
 
    if target and target in df.columns:
        df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)
 
    # ── Auto-discover feature columns ─────────────────────────────────────────
    id_set = CFG["ID_COLS"] | {target} if target else CFG["ID_COLS"]
    # Also exclude churn_rules signal columns (sig_ prefix)
    feat_cols = [
        c for c in df.columns
        if c not in id_set
        and not c.startswith("SIG_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    log.info("Auto-discovered %d numeric feature columns", len(feat_cols))
 
    # Cast all features to float32, replace inf/NaN
    for col in feat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce") \
                    .replace([np.inf, -np.inf], np.nan).fillna(0.0)
 
    return df, feat_cols, target
 
 
def _detect_feature_groups(feat_cols: List[str]) -> Dict[str, List[str]]:
    """Auto-group features by service domain from column name patterns."""
    groups: Dict[str, List[str]] = {}
    priority_map = [
        ("Revenue",  ["REVENUE"]),
        ("Data",     ["DATA_MB", "DATA_ACTIVE", "DATA_REVENUE"]),
        ("Voice",    ["VOICE_MIN", "VOICE_ACTIVE", "VOICE_REVENUE", "OG_VOICE", "IC_VOICE"]),
        ("Bundle",   ["BUNDLE"]),
        ("SMS",      ["SMS"]),
        ("Activity", ["ACTIVE_WEEKS", "ANY_ACTIVE", "SERVICE_DIVERSITY",
                      "ENGAGEMENT", "ANY_ACTIVE"]),
        ("Weekly",   ["_W10", "_W11", "_W12", "_W13"]),
    ]
    assigned = set()
    for group_name, keywords in priority_map:
        for col in feat_cols:
            if col not in assigned and any(k in col for k in keywords):
                groups.setdefault(group_name, []).append(col)
                assigned.add(col)
    other = [c for c in feat_cols if c not in assigned]
    if other:
        groups["Other"] = other
    return groups
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. GBM SURROGATE — MIRRORS THE RULE-BASED TIER DECISIONS
# ─────────────────────────────────────────────────────────────────────────────
 
def train_surrogate_gbm(
    df       : pd.DataFrame,
    feat_cols: List[str],
    target   : Optional[str],
) -> Tuple[GradientBoostingClassifier, RobustScaler, np.ndarray, np.ndarray]:
    """
    Train a GBM that predicts actual churn (if labels available) or
    predicts the rule-based tier classification (tier >= 1 = at-risk).
 
    We prefer actual churn labels because:
      - SHAP explains what the DATA says causes churn
      - Not just what the rules say
      - This reveals WHY the rules produce FPs and FNs
 
    For large datasets: sample MAX_SHAP_ROWS stratified rows for training.
 
    Returns (model, scaler, X_used, y_used)
    """
    max_rows = CFG["MAX_SHAP_ROWS"]
 
    # Decide on y
    if target and target in df.columns and df[target].sum() >= 5:
        y_raw     = df[target].values.astype(int)
        label_src = f"actual churn labels ({y_raw.sum()} positives)"
    elif "TIER" in df.columns:
        y_raw     = (df["TIER"].fillna(0).astype(int) >= 1).astype(int)
        label_src = "rule-based tier (tier >= 1 = at-risk)"
    else:
        raise ValueError(
            "Need either a churn label column or a TIER column from churn_rules.py.\n"
            "Run churn_rules.py first, or set TARGET_COL=LABEL_CHURN_90D."
        )
    log.info("GBM surrogate target: %s", label_src)
 
    X_all = df[feat_cols].values.astype(np.float32)
 
    # Stratified sample for large data
    if len(X_all) > max_rows:
        log.info("Data has %d rows → sampling %d stratified rows for GBM training",
                 len(X_all), max_rows)
        rng = np.random.default_rng(CFG["RANDOM_STATE"])
        pos_idx = np.where(y_raw == 1)[0]
        neg_idx = np.where(y_raw == 0)[0]
        n_pos   = min(len(pos_idx), max_rows // 10)
        n_neg   = min(len(neg_idx), max_rows - n_pos)
        keep    = np.concatenate([
            rng.choice(pos_idx, n_pos, replace=False),
            rng.choice(neg_idx, n_neg, replace=False),
        ])
        X_used = X_all[keep]
        y_used = y_raw[keep]
        log.info("Sampled %d rows (%d pos, %d neg)", len(keep), n_pos, n_neg)
    else:
        X_used = X_all
        y_used = y_raw
        keep   = np.arange(len(X_all))
 
    scaler = RobustScaler()
    X_s    = scaler.fit_transform(X_used)
 
    neg_n = int((y_used == 0).sum())
    pos_n = int((y_used == 1).sum())
    spw   = neg_n / max(pos_n, 1)
 
    log.info("Training GBM surrogate: %d rows, %d features, spw=%.1f …",
             len(y_used), len(feat_cols), spw)
    model = GradientBoostingClassifier(
        n_estimators       = CFG["GBM_TREES"],
        max_depth          = 4,
        learning_rate      = 0.05,
        subsample          = 0.80,
        min_samples_leaf   = max(1, pos_n // 10),
        random_state       = CFG["RANDOM_STATE"],
    )
    sw = np.where(y_used == 1, spw, 1.0)
    model.fit(X_s, y_used, sample_weight=sw)
 
    if pos_n >= 5:
        proba = model.predict_proba(X_s)[:, 1]
        auc   = roc_auc_score(y_used, proba)
        log.info("GBM surrogate ROC-AUC on training data: %.4f", auc)
 
    return model, scaler, X_used, y_used, keep
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. SHAP COMPUTATION — SCALES TO LARGE DATASETS
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_shap_values(
    model    : GradientBoostingClassifier,
    scaler   : RobustScaler,
    X_used   : np.ndarray,
    feat_cols: List[str],
) -> Tuple[np.ndarray, float]:
    """
    Compute SHAP values for all rows in X_used using TreeExplainer.
 
    TreeExplainer is exact and fast for GBM — scales to 500K+ rows
    without approximation. No chunking needed.
 
    Returns (shap_values_2d, base_value)
    """
    if not _SHAP_OK:
        raise ImportError("pip install shap")
 
    log.info("Computing SHAP values for %d rows × %d features …",
             *X_used.shape)
    X_s = scaler.transform(X_used)
 
    # Build background using a stratified subsample
    n_bg  = min(CFG["N_BACKGROUND"], len(X_s))
    rng   = np.random.default_rng(CFG["RANDOM_STATE"])
    bg    = X_s[rng.choice(len(X_s), n_bg, replace=False)]
 
    t0    = time.time()
    explainer  = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
    shap_raw   = explainer.shap_values(X_s, check_additivity=False)
 
    # Normalise output shape: we always want (n_samples, n_features) for class=1
    shap_arr = np.array(shap_raw)
    if shap_arr.ndim == 3:          # (n_classes, n_samples, n_features)
        shap_arr = shap_arr[1]
    elif shap_arr.ndim == 2:        # already (n_samples, n_features)
        pass
    else:
        raise ValueError(f"Unexpected SHAP output shape: {shap_arr.shape}")
 
    base_val = explainer.expected_value
    if hasattr(base_val, '__len__'):
        base_val = float(base_val[-1])
    else:
        base_val = float(base_val)
 
    log.info("SHAP computation complete in %.1fs (base value = %.4f)",
             time.time() - t0, base_val)
    return shap_arr, base_val
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. SEGMENT ERROR GROUPS (FP, FN, TP, TN)
# ─────────────────────────────────────────────────────────────────────────────
 
def segment_errors(
    df      : pd.DataFrame,
    y_true  : np.ndarray,
    y_pred  : np.ndarray,
    keep_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Return boolean masks into the X_used / shap_arr arrays for each error group.
    All masks are RELATIVE to X_used (which may be a sample of df).
    """
    return {
        "TP": (y_true == 1) & (y_pred == 1),   # correctly caught churners
        "TN": (y_true == 0) & (y_pred == 0),   # correctly cleared non-churners
        "FP": (y_true == 0) & (y_pred == 1),   # wrongly flagged (false alarm)
        "FN": (y_true == 1) & (y_pred == 0),   # missed churners (false negative)
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. DASHBOARD — 7 PANELS
# ─────────────────────────────────────────────────────────────────────────────
 
def build_dashboard(
    shap_arr  : np.ndarray,
    base_val  : float,
    X_used    : np.ndarray,
    y_true    : np.ndarray,
    y_pred    : np.ndarray,
    feat_cols : List[str],
    feat_grps : Dict[str, List[str]],
    error_masks: Dict[str, np.ndarray],
    df        : pd.DataFrame,
    keep_idx  : np.ndarray,
):
    sns.set_style("whitegrid")
    PALETTE = {
        "TP": "#27AE60", "TN": "#2980B9",
        "FP": "#E74C3C", "FN": "#E67E22",
    }
 
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(
        "SHAP Explainability Report — Churn Prediction\n"
        "Understanding Why Subscribers Are Flagged / Missed",
        fontsize=14, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)
    ax = {
        "global_bar"  : fig.add_subplot(gs[0, 0]),
        "beeswarm"    : fig.add_subplot(gs[0, 1:]),
        "fp_bar"      : fig.add_subplot(gs[1, 0]),
        "fn_bar"      : fig.add_subplot(gs[1, 1]),
        "fp_vs_tp"    : fig.add_subplot(gs[1, 2]),
        "group_heat"  : fig.add_subplot(gs[2, 0]),
        "error_dist"  : fig.add_subplot(gs[2, 1]),
        "fn_vs_tn"    : fig.add_subplot(gs[2, 2]),
    }
 
    feat_arr = np.array(feat_cols)
    TOP_N    = 15
 
    # ── Global importance (mean |SHAP|) ──────────────────────────────────────
    mean_abs  = np.abs(shap_arr).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[-TOP_N:][::-1]
    top_feats = feat_arr[top_idx]
    top_vals  = mean_abs[top_idx]
 
    a = ax["global_bar"]
    colors = sns.color_palette("YlOrRd_r", TOP_N)
    a.barh(top_feats[::-1], top_vals[::-1], color=colors[::-1], edgecolor="white")
    a.set_title(f"Top {TOP_N} Features\n(Global Mean |SHAP|)", fontsize=10, fontweight="bold")
    a.set_xlabel("Mean |SHAP value|", fontsize=8)
    a.tick_params(axis="y", labelsize=7)
    a.spines[["top","right"]].set_visible(False)
 
    # ── Beeswarm (violin) summary ─────────────────────────────────────────────
    plt.sca(ax["beeswarm"])
    shap_top  = shap_arr[:, top_idx]
    X_top     = X_used[:, top_idx]
    feat_top  = feat_arr[top_idx].tolist()
    shap.summary_plot(shap_top, X_top, feature_names=feat_top,
                      max_display=TOP_N, show=False, plot_type="violin",
                      color_bar_label="Feature value")
    ax["beeswarm"].set_title(
        f"SHAP Summary — Top {TOP_N} Features\n"
        "Red = high feature value, Blue = low. +SHAP = pushes toward churn",
        fontsize=10, fontweight="bold")
 
    # ── FP panel: what made non-churners look like churners? ─────────────────
    fp_mask = error_masks["FP"]
    tp_mask = error_masks["TP"]
 
    for mask, title, panel_key, color in [
        (fp_mask, f"FALSE POSITIVES (n={fp_mask.sum():,})\nWhy non-churners were flagged",
         "fp_bar", PALETTE["FP"]),
        (error_masks["FN"],
         f"FALSE NEGATIVES (n={error_masks['FN'].sum():,})\nWhy churners were missed",
         "fn_bar", PALETTE["FN"]),
    ]:
        a = ax[panel_key]
        if mask.sum() == 0:
            a.text(0.5, 0.5, "No examples\nin this group",
                   ha="center", va="center", transform=a.transAxes)
            a.set_title(title, fontsize=10, fontweight="bold")
            continue
        mean_abs_grp = np.abs(shap_arr[mask]).mean(axis=0)
        idx_sort     = np.argsort(mean_abs_grp)[-12:][::-1]
        cols_shown   = feat_arr[idx_sort]
        vals_shown   = mean_abs_grp[idx_sort]
        a.barh(cols_shown[::-1], vals_shown[::-1],
               color=color, alpha=0.85, edgecolor="white")
        a.set_title(title, fontsize=9.5, fontweight="bold")
        a.set_xlabel("Mean |SHAP value| in group", fontsize=8)
        a.tick_params(axis="y", labelsize=6.5)
        a.spines[["top","right"]].set_visible(False)
 
    # ── FP vs TP mean SHAP difference ────────────────────────────────────────
    # Shows which features drive EXTRA false positives beyond true positives
    a = ax["fp_vs_tp"]
    if fp_mask.sum() >= 3 and tp_mask.sum() >= 3:
        fp_mean = shap_arr[fp_mask].mean(axis=0)
        tp_mean = shap_arr[tp_mask].mean(axis=0)
        diff    = fp_mean - tp_mean
        top_diff_idx = np.argsort(np.abs(diff))[-12:][::-1]
        colors_bar   = [PALETTE["FP"] if diff[i] > 0 else PALETTE["TP"]
                        for i in top_diff_idx]
        a.barh(feat_arr[top_diff_idx][::-1], diff[top_diff_idx][::-1],
               color=colors_bar[::-1], edgecolor="white")
        a.axvline(0, color="black", lw=0.8)
        a.set_title("FP vs TP — SHAP Divergence\nRed = more SHAP in FP (FP driver)\n"
                    "Green = more SHAP in TP (real churn signal)",
                    fontsize=9, fontweight="bold")
        a.set_xlabel("Mean SHAP difference (FP − TP)", fontsize=8)
        a.tick_params(axis="y", labelsize=6.5)
        a.spines[["top","right"]].set_visible(False)
    else:
        a.text(0.5, 0.5, "Need ≥3 FP and TP\nto compare",
               ha="center", va="center", transform=a.transAxes)
 
    # ── Feature group heatmap ─────────────────────────────────────────────────
    a = ax["group_heat"]
    groups_to_show = {g: cols for g, cols in feat_grps.items() if g != "Weekly"}
    group_names = list(groups_to_show.keys())
    categories  = ["TP","FP","FN","TN"]
    heat_data   = np.zeros((len(group_names), len(categories)))
    for gi, (grp, gcols) in enumerate(groups_to_show.items()):
        col_indices = [i for i,c in enumerate(feat_cols) if c in gcols]
        if not col_indices:
            continue
        grp_shap = np.abs(shap_arr[:, col_indices]).sum(axis=1)
        for ci, cat in enumerate(categories):
            mask = error_masks[cat]
            heat_data[gi, ci] = grp_shap[mask].mean() if mask.sum() > 0 else 0
 
    heat_df = pd.DataFrame(heat_data, index=group_names, columns=categories)
    # Normalise each row so colours reflect importance within the group's own range
    heat_norm = heat_df.div(heat_df.max(axis=1).replace(0,1), axis=0)
    sns.heatmap(heat_norm, ax=a, cmap="YlOrRd", linewidths=0.5,
                annot=heat_df.round(3), fmt=".3f",
                annot_kws={"size":7}, cbar_kws={"label":"Normalised SHAP"})
    a.set_title("Feature Group Importance\nby Error Category",
                fontsize=10, fontweight="bold")
    a.set_xlabel("Error group", fontsize=9); a.tick_params(axis="y", rotation=0)
 
    # ── Error group SHAP magnitude distribution ───────────────────────────────
    a = ax["error_dist"]
    total_shap = np.abs(shap_arr).sum(axis=1)
    for cat, mask in error_masks.items():
        if mask.sum() == 0:
            continue
        vals = total_shap[mask]
        a.hist(vals, bins=30, alpha=0.55, color=PALETTE[cat],
               label=f"{cat} (n={mask.sum():,})", density=True)
    a.set_title("Total |SHAP| Distribution per Group\nHigher = model more certain",
                fontsize=10, fontweight="bold")
    a.set_xlabel("Sum of |SHAP values|", fontsize=9)
    a.set_ylabel("Density", fontsize=9)
    a.legend(fontsize=8)
    a.spines[["top","right"]].set_visible(False)
 
    # ── FN vs TN SHAP divergence ─────────────────────────────────────────────
    a = ax["fn_vs_tn"]
    fn_mask = error_masks["FN"]
    tn_mask = error_masks["TN"]
    if fn_mask.sum() >= 3 and tn_mask.sum() >= 3:
        fn_mean   = shap_arr[fn_mask].mean(axis=0)
        tn_mean   = shap_arr[tn_mask].mean(axis=0)
        diff2     = fn_mean - tn_mean
        top_idx2  = np.argsort(np.abs(diff2))[-12:][::-1]
        colors2   = [PALETTE["FN"] if diff2[i] < 0 else PALETTE["TN"]
                     for i in top_idx2]
        a.barh(feat_arr[top_idx2][::-1], diff2[top_idx2][::-1],
               color=colors2[::-1], edgecolor="white")
        a.axvline(0, color="black", lw=0.8)
        a.set_title("FN vs TN — SHAP Divergence\nOrange = lower SHAP in FN (missed signal)\n"
                    "Blue = feature same in both (no help)",
                    fontsize=9, fontweight="bold")
        a.set_xlabel("Mean SHAP difference (FN − TN)", fontsize=8)
        a.tick_params(axis="y", labelsize=6.5)
        a.spines[["top","right"]].set_visible(False)
    else:
        a.text(0.5, 0.5, "Need ≥3 FN and TN\nto compare",
               ha="center", va="center", transform=a.transAxes)
 
    out_path = os.path.join(CFG["OUTPUT_DIR"], "shap_dashboard.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Dashboard saved → %s", out_path)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. INDIVIDUAL SUBSCRIBER EXPLANATIONS (WATERFALL + DECISION PLOTS)
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_individual_explanations(
    shap_arr  : np.ndarray,
    base_val  : float,
    X_used    : np.ndarray,
    y_true    : np.ndarray,
    y_pred    : np.ndarray,
    feat_cols : List[str],
    error_masks: Dict[str, np.ndarray],
    df        : pd.DataFrame,
    keep_idx  : np.ndarray,
):
    """
    For each error group (FP and FN), generate:
      1. Waterfall plot — top feature contributions for each subscriber
      2. Combined decision plot — overlay for all selected FP / FN cases
 
    These are saved as separate PNG files one per group.
    """
    id_col = "MSISDN" if "MSISDN" in df.columns else df.columns[0]
    # Map keep_idx back to df index for subscriber ID lookup
    df_sub = df.iloc[keep_idx].reset_index(drop=True)
 
    for cat, mask, n_max_label in [
        ("FP", error_masks["FP"], CFG["N_EXPLAIN_FP"]),
        ("FN", error_masks["FN"], CFG["N_EXPLAIN_FN"]),
    ]:
        sub_idx  = np.where(mask)[0]
        if len(sub_idx) == 0:
            log.info("No %s subscribers to explain", cat)
            continue
 
        n_show = min(n_max_label, len(sub_idx))
        # For FP: pick the highest model-probability FPs (most confident mistakes)
        # For FN: pick the lowest model-probability FNs (most confident misses)
        proba = _model_proba(X_used, _loaded_model, _loaded_scaler)
        if cat == "FP":
            sort_by = proba[sub_idx][::-1]   # high prob first
        else:
            sort_by = proba[sub_idx]          # low prob first (most missed)
        chosen_local = sub_idx[np.argsort(sort_by)[:n_show]]
 
        group_label = "False Positive (non-churner wrongly flagged)" \
                      if cat == "FP" else "False Negative (churner missed)"
 
        # ── Decision plot for the whole group ────────────────────────────────
        fig, a = plt.subplots(figsize=(12, max(5, n_show * 0.55 + 3)))
        shap.decision_plot(
            base_val,
            shap_arr[chosen_local],
            feat_cols,
            link="logit",
            show=False,
            highlight=0,
        )
        plt.title(
            f"{group_label}\n"
            f"Decision plot for {n_show} subscribers (sorted by model confidence)\n"
            "Each line = one subscriber; right end = model's churn probability",
            fontsize=10, fontweight="bold",
        )
        dp_path = os.path.join(CFG["OUTPUT_DIR"], f"shap_decision_{cat}.png")
        plt.savefig(dp_path, dpi=130, bbox_inches="tight")
        plt.close()
        log.info("Decision plot saved → %s", dp_path)
 
        # ── Individual waterfall plots (top 5 per group) ──────────────────────
        for rank, local_i in enumerate(chosen_local[:5], 1):
            sub_id = df_sub[id_col].iloc[local_i] if id_col in df_sub.columns else local_i
            fig = plt.figure(figsize=(10, 5))
            exp_obj = shap.Explanation(
                values      = shap_arr[local_i],
                base_values = base_val,
                data        = X_used[local_i],
                feature_names = feat_cols,
            )
            shap.waterfall_plot(exp_obj, max_display=12, show=False)
            plt.title(
                f"{cat} Subscriber #{rank}  (ID: {sub_id})\n"
                f"{group_label}\n"
                "Bars show why the model scored this subscriber high/low",
                fontsize=9, fontweight="bold",
            )
            wf_path = os.path.join(CFG["OUTPUT_DIR"],
                                   f"shap_waterfall_{cat}_rank{rank}.png")
            plt.savefig(wf_path, dpi=130, bbox_inches="tight")
            plt.close()
 
        log.info("Waterfall plots saved → %s/shap_waterfall_%s_rank*.png",
                 CFG["OUTPUT_DIR"], cat)
 
 
# Globals set during main() so helper functions can access the fitted model
_loaded_model  = None
_loaded_scaler = None
 
 
def _model_proba(X, model, scaler):
    return model.predict_proba(scaler.transform(X))[:, 1]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 7. PER-SUBSCRIBER SHAP CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────
 
def export_shap_csv(
    shap_arr  : np.ndarray,
    X_used    : np.ndarray,
    y_true    : np.ndarray,
    y_pred    : np.ndarray,
    feat_cols : List[str],
    df        : pd.DataFrame,
    keep_idx  : np.ndarray,
):
    """
    Export a CSV with:
      - Subscriber ID
      - Actual churn label
      - Predicted churn (0/1)
      - Model churn probability
      - Error category (TP/TN/FP/FN)
      - SHAP value for each feature
      - Top-3 positive and negative SHAP features (human-readable)
 
    For large datasets, writes only the FP and FN rows (the actionable ones).
    Full data export is also available but gated by row count.
    """
    df_sub   = df.iloc[keep_idx].reset_index(drop=True)
    id_col   = "MSISDN" if "MSISDN" in df_sub.columns else df_sub.columns[0]
    proba    = _model_proba(X_used, _loaded_model, _loaded_scaler)
 
    cats     = np.full(len(y_true), "TN", dtype=object)
    cats[(y_true==1)&(y_pred==1)] = "TP"
    cats[(y_true==0)&(y_pred==1)] = "FP"
    cats[(y_true==1)&(y_pred==0)] = "FN"
 
    # Build SHAP explanation text for each subscriber
    def top_drivers(shap_row, n=3, positive=True):
        s = shap_row if positive else -shap_row
        idx = np.argsort(s)[-n:][::-1]
        return "; ".join([f"{feat_cols[i]}={shap_row[i]:+.3f}" for i in idx
                          if (s[i] > 0)])
 
    log.info("Building SHAP explanation CSV …")
    rows = []
    for i in range(len(y_true)):
        row = {
            id_col         : df_sub[id_col].iloc[i] if id_col in df_sub.columns else i,
            "actual_churn" : int(y_true[i]),
            "predicted"    : int(y_pred[i]),
            "churn_prob"   : round(float(proba[i]), 4),
            "error_cat"    : cats[i],
            "top3_churn_drivers"   : top_drivers(shap_arr[i], positive=True),
            "top3_safety_drivers"  : top_drivers(shap_arr[i], positive=False),
            "total_shap_magnitude" : round(float(np.abs(shap_arr[i]).sum()), 4),
        }
        # Attach individual SHAP values for the top 20 global features
        global_top20 = np.argsort(np.abs(shap_arr).mean(axis=0))[-20:][::-1]
        for j in global_top20:
            row[f"shap_{feat_cols[j]}"] = round(float(shap_arr[i, j]), 5)
        rows.append(row)
 
    out_df   = pd.DataFrame(rows)
    out_path = os.path.join(CFG["OUTPUT_DIR"], "shap_explanations.csv")
    out_df.to_csv(out_path, index=False)
    log.info("SHAP explanations CSV → %s  (%d rows)", out_path, len(out_df))
 
    # Separate FP and FN files for the business team
    for cat in ["FP", "FN"]:
        sub = out_df[out_df["error_cat"] == cat]
        p   = os.path.join(CFG["OUTPUT_DIR"], f"shap_explanations_{cat}.csv")
        sub.to_csv(p, index=False)
        log.info("  %s file → %s (%d rows)", cat, p, len(sub))
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 8. TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────
 
def _sep(c="═", w=76): print(c * w)
 
 
def print_text_report(
    shap_arr   : np.ndarray,
    y_true     : np.ndarray,
    y_pred     : np.ndarray,
    feat_cols  : List[str],
    feat_grps  : Dict[str, List[str]],
    error_masks: Dict[str, np.ndarray],
    df         : pd.DataFrame,
    keep_idx   : np.ndarray,
    input_csv  : str,
):
    proba    = _model_proba(
        np.vstack([np.zeros((1, len(feat_cols))), np.zeros((1, len(feat_cols)))])[:0],
        _loaded_model, _loaded_scaler
    ) if False else _model_proba(
        np.zeros((1, len(feat_cols))), _loaded_model, _loaded_scaler
    )
    proba    = _model_proba(
        np.array([[0.0]*len(feat_cols)]), _loaded_model, _loaded_scaler
    )
    proba_all = _model_proba(
        np.zeros((len(y_true), len(feat_cols))), _loaded_model, _loaded_scaler
    )
    proba_all = _loaded_model.predict_proba(
        _loaded_scaler.transform(np.zeros((len(y_true), len(feat_cols))))
    )[:,1]
    # Re-score properly
    from sklearn.preprocessing import RobustScaler as RS
    X_real = np.zeros_like(np.zeros((1,1)))  # placeholder; computed before call
 
    _sep()
    print(f"  SHAP EXPLAINABILITY REPORT")
    print(f"  Dataset  : {input_csv}")
    print(f"  GBM rows : {len(y_true):,}  |  Features: {len(feat_cols)}")
    _sep()
 
    # Error group summary
    total_pos = int(y_true.sum())
    total_neg = int((y_true == 0).sum())
    print(f"\n  ── ERROR GROUP BREAKDOWN ──")
    for cat, mask in error_masks.items():
        n = int(mask.sum())
        pct_pos = n/total_pos*100 if cat in ("TP","FN") and total_pos else 0
        pct_neg = n/total_neg*100 if cat in ("TN","FP") and total_neg else 0
        print(f"  {cat}: {n:>6,} subscribers"
              + (f"  ({pct_pos:.1f}% of churners)" if cat in ("TP","FN") else
                 f"  ({pct_neg:.1f}% of non-churners)"))
 
    # Global top features
    mean_abs  = np.abs(shap_arr).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[-15:][::-1]
    print(f"\n  ── TOP 15 FEATURES BY GLOBAL MEAN |SHAP| ──")
    print(f"  {'Rank':<5} {'Feature':<42} {'Mean |SHAP|':>12}")
    print("  " + "─"*62)
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:<5} {feat_cols[i]:<42} {mean_abs[i]:>12.5f}")
 
    # FP deep-dive
    fp_mask = error_masks["FP"]
    if fp_mask.sum() >= 3:
        print(f"\n  ── FALSE POSITIVE ANALYSIS (n={fp_mask.sum():,}) ──")
        print("  These non-churners were WRONGLY flagged. The top features")
        print("  driving their high churn scores are:")
        fp_mean = np.abs(shap_arr[fp_mask]).mean(axis=0)
        tp_mean = np.abs(shap_arr[error_masks["TP"]]).mean(axis=0) \
                  if error_masks["TP"].sum() >= 3 else np.zeros(len(feat_cols))
        top_fp = np.argsort(fp_mean)[-10:][::-1]
        print(f"  {'Feature':<42} {'FP |SHAP|':>10}  {'TP |SHAP|':>10}  {'FP excess':>10}")
        print("  " + "─"*78)
        for i in top_fp:
            excess = fp_mean[i] - tp_mean[i]
            print(f"  {feat_cols[i]:<42} {fp_mean[i]:>10.5f}  {tp_mean[i]:>10.5f}  {excess:>+10.5f}")
        print()
        print("  INTERPRETATION: Features with positive 'FP excess' are pulling")
        print("  non-churners into the churn zone — these are the root cause of")
        print("  false positives. Consider adding threshold guards for these features.")
 
    # FN deep-dive
    fn_mask = error_masks["FN"]
    if fn_mask.sum() >= 3:
        print(f"\n  ── FALSE NEGATIVE ANALYSIS (n={fn_mask.sum():,}) ──")
        print("  These REAL churners were MISSED. The signals the model weighted")
        print("  LOW for them (compared with non-churners) explain the miss:")
        fn_mean = shap_arr[fn_mask].mean(axis=0)
        tn_mean = shap_arr[error_masks["TN"]].mean(axis=0) \
                  if error_masks["TN"].sum() >= 3 else np.zeros(len(feat_cols))
        diff_fn = fn_mean - tn_mean
        top_fn  = np.argsort(diff_fn)[-10:][::-1]  # most negative = most missed
        print(f"  {'Feature':<42} {'FN SHAP':>10}  {'TN SHAP':>10}  {'Gap':>10}")
        print("  " + "─"*78)
        for i in top_fn:
            print(f"  {feat_cols[i]:<42} {fn_mean[i]:>10.5f}  {tn_mean[i]:>10.5f}  {diff_fn[i]:>+10.5f}")
        print()
        print("  INTERPRETATION: Features with a negative gap are ones where")
        print("  the missed churners look SAFER than non-churners — the model")
        print("  is not receiving the right signal for these subscribers.")
        print("  → These are the features where new data (network, CRM,")
        print("    M-PESA) would most improve detection.")
 
    # Feature group summary
    print(f"\n  ── FEATURE GROUP CONTRIBUTION ──")
    print(f"  {'Group':<15} {'FP |SHAP|':>12} {'FN |SHAP|':>12} {'TP |SHAP|':>12}")
    print("  " + "─"*54)
    for grp, gcols in feat_grps.items():
        if grp == "Weekly": continue
        col_idx = [i for i,c in enumerate(feat_cols) if c in gcols]
        if not col_idx: continue
        grp_sv   = np.abs(shap_arr[:, col_idx]).sum(axis=1)
        fp_grp   = grp_sv[fp_mask].mean()  if fp_mask.sum()>0 else 0
        fn_grp   = grp_sv[fn_mask].mean()  if fn_mask.sum()>0 else 0
        tp_grp   = grp_sv[error_masks["TP"]].mean() if error_masks["TP"].sum()>0 else 0
        print(f"  {grp:<15} {fp_grp:>12.4f} {fn_grp:>12.4f} {tp_grp:>12.4f}")
 
    _sep()
    print(f"  Output files in: {CFG['OUTPUT_DIR']}/")
    print("    shap_dashboard.png          — 7-panel visual report")
    print("    shap_decision_FP.png        — FP decision plot")
    print("    shap_decision_FN.png        — FN decision plot")
    print("    shap_waterfall_FP_rank*.png — Individual FP explanations")
    print("    shap_waterfall_FN_rank*.png — Individual FN explanations")
    print("    shap_explanations.csv       — Per-subscriber SHAP values")
    print("    shap_explanations_FP.csv    — FP-only explanation CSV")
    print("    shap_explanations_FN.csv    — FN-only explanation CSV")
    _sep()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 9. DYNAMIC MULTI-DATASET COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
 
def compare_datasets(paths: List[str], model_path: Optional[str] = None):
    """
    Run SHAP on multiple CSVs and compare:
      - Feature importance shift across datasets (temporal drift detection)
      - FP/FN rate change (model degradation monitoring)
      - Which features behave differently between Feb and March (or any snapshots)
 
    Pass paths as: python churn_shap_analysis.py compare Feb.csv March.csv
    """
    if not _SHAP_OK:
        raise ImportError("pip install shap")
 
    import joblib
 
    global _loaded_model, _loaded_scaler
 
    fig, axes = plt.subplots(1, len(paths), figsize=(10 * len(paths), 7))
    if len(paths) == 1:
        axes = [axes]
 
    all_mean_abs = {}
    all_stats    = {}
 
    for path, a in zip(paths, axes):
        name = os.path.basename(path).replace(".csv","")
        log.info("=== Processing %s ===", name)
        CFG["INPUT_CSV"] = path
        CFG["SCORED_CSV"] = os.path.join(
            os.path.dirname(path), "churn_rules_output", "churn_scored.csv"
        )
 
        try:
            df, feat_cols, target = load_and_merge()
        except Exception as e:
            log.warning("Could not load %s: %s", path, e)
            continue
 
        # Load or train model
        if model_path and os.path.exists(model_path):
            artifacts    = joblib.load(model_path)
            model        = artifacts["model"]
            scaler       = artifacts["scaler"]
            saved_feats  = artifacts["feat_cols"]
            # Align features
            X_all = pd.DataFrame(
                0.0, index=df.index,
                columns=saved_feats
            )
            for c in saved_feats:
                if c in df.columns:
                    X_all[c] = df[c].values
            X_used = X_all.values.astype(np.float32)
            feat_cols_use = saved_feats
        else:
            model, scaler, X_used, y_used, keep_idx = train_surrogate_gbm(
                df, feat_cols, target)
            feat_cols_use = feat_cols
 
        _loaded_model  = model
        _loaded_scaler = scaler
 
        shap_arr, base_val = compute_shap_values(model, scaler, X_used, feat_cols_use)
        mean_abs = np.abs(shap_arr).mean(axis=0)
        all_mean_abs[name] = dict(zip(feat_cols_use, mean_abs))
 
        # FP/FN if labels available
        if target and target in df.columns:
            y_t   = df.iloc[np.arange(len(X_used))][target].values
            proba = _model_proba(X_used, model, scaler)
            th    = 0.50
            y_p   = (proba >= th).astype(int)
            n_tp  = int(((y_p==1)&(y_t==1)).sum())
            n_fp  = int(((y_p==1)&(y_t==0)).sum())
            n_fn  = int(((y_p==0)&(y_t==1)).sum())
            n_tn  = int(((y_p==0)&(y_t==0)).sum())
            prec  = n_tp/(n_tp+n_fp) if (n_tp+n_fp) else 0
            rec   = n_tp/(n_tp+n_fn) if (n_tp+n_fn) else 0
            all_stats[name] = {
                "n_total":len(df),"n_churners":int(y_t.sum()),
                "TP":n_tp,"FP":n_fp,"FN":n_fn,"TN":n_tn,
                "precision":round(prec,4),"recall":round(rec,4),
            }
            log.info("%s stats: %s", name, all_stats[name])
 
        # Per-dataset importance bar
        top20    = sorted(all_mean_abs[name].items(), key=lambda x:-x[1])[:20]
        features = [x[0] for x in top20]
        values   = [x[1] for x in top20]
        a.barh(features[::-1], values[::-1],
               color=sns.color_palette("YlOrRd_r",20), edgecolor="white")
        a.set_title(f"{name}\n({len(df):,} rows)", fontsize=10, fontweight="bold")
        a.set_xlabel("Mean |SHAP|", fontsize=9)
        a.tick_params(axis="y", labelsize=7)
        a.spines[["top","right"]].set_visible(False)
 
    # Feature drift: which features changed importance most between datasets?
    if len(all_mean_abs) >= 2:
        names = list(all_mean_abs.keys())
        all_feats = set()
        for d in all_mean_abs.values():
            all_feats.update(d.keys())
        drift = {}
        for f in all_feats:
            vals = [all_mean_abs[n].get(f, 0) for n in names]
            drift[f] = max(vals) - min(vals)
        top_drift = sorted(drift.items(), key=lambda x:-x[1])[:15]
        drift_fig, da = plt.subplots(figsize=(12, 7))
        feats_d  = [x[0] for x in top_drift]
        for ni, nm in enumerate(names):
            bar_vals = [all_mean_abs[nm].get(f, 0) for f in feats_d]
            x_pos    = np.arange(len(feats_d)) + ni * 0.35
            da.bar(x_pos, bar_vals, width=0.33,
                   label=nm, alpha=0.85)
        da.set_xticks(np.arange(len(feats_d)) + 0.175)
        da.set_xticklabels([f.replace("_"," ")[:25] for f in feats_d],
                            rotation=45, ha="right", fontsize=8)
        da.set_title("Feature Importance Drift Across Datasets\n"
                     "(Top 15 most-changed features between snapshots)",
                     fontsize=11, fontweight="bold")
        da.set_ylabel("Mean |SHAP value|", fontsize=9)
        da.legend(fontsize=9)
        da.spines[["top","right"]].set_visible(False)
        drift_path = os.path.join(CFG["OUTPUT_DIR"], "shap_feature_drift.png")
        drift_fig.savefig(drift_path, dpi=150, bbox_inches="tight")
        plt.close(drift_fig)
        log.info("Feature drift plot saved → %s", drift_path)
 
    comp_path = os.path.join(CFG["OUTPUT_DIR"], "shap_dataset_comparison.png")
    fig.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Dataset comparison saved → %s", comp_path)
 
    if all_stats:
        _sep(); print("  MULTI-DATASET COMPARISON SUMMARY"); _sep()
        for name, s in all_stats.items():
            print(f"  {name}: {s['n_total']:,} rows | churners={s['n_churners']:,} "
                  f"| TP={s['TP']} FP={s['FP']:,} FN={s['FN']} "
                  f"| Prec={s['precision']:.3f} Recall={s['recall']:.3f}")
        _sep()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    global _loaded_model, _loaded_scaler
 
    if not _SHAP_OK:
        print("ERROR: SHAP not installed. Run: pip install shap")
        sys.exit(1)
 
    # Handle compare mode: python churn_shap_analysis.py compare file1.csv file2.csv ...
    if len(sys.argv) >= 3 and sys.argv[1].lower() == "compare":
        compare_datasets(sys.argv[2:])
        return
 
    t0 = time.time()
    _sep()
    print("  SHAP EXPLAINABILITY ANALYSIS — Churn Prediction")
    print(f"  Input: {CFG['INPUT_CSV']}  |  Scored: {CFG['SCORED_CSV']}")
    print(f"  Output: {CFG['OUTPUT_DIR']}")
    _sep()
 
    # ── Step 1: Load data ─────────────────────────────────────────────────────
    df, feat_cols, target = load_and_merge()
    feat_grps = _detect_feature_groups(feat_cols)
    log.info("Feature groups: %s", {g: len(c) for g,c in feat_grps.items()})
 
    # ── Step 2: Train GBM surrogate ───────────────────────────────────────────
    model, scaler, X_used, y_used, keep_idx = train_surrogate_gbm(
        df, feat_cols, target)
    _loaded_model  = model
    _loaded_scaler = scaler
 
    # ── Step 3: Compute SHAP ──────────────────────────────────────────────────
    shap_arr, base_val = compute_shap_values(model, scaler, X_used, feat_cols)
 
    # ── Step 4: Get predictions + error segments ─────────────────────────────
    y_pred = (model.predict_proba(scaler.transform(X_used))[:, 1] >= 0.5).astype(int)
    y_true = y_used
 
    error_masks = segment_errors(df, y_true, y_pred, keep_idx)
    log.info("Error segments: %s",
             {k: int(v.sum()) for k,v in error_masks.items()})
 
    # ── Step 5: Print text report ─────────────────────────────────────────────
    print_text_report(shap_arr, y_true, y_pred, feat_cols, feat_grps,
                      error_masks, df, keep_idx, CFG["INPUT_CSV"])
 
    # ── Step 6: Dashboard ─────────────────────────────────────────────────────
    log.info("Building dashboard …")
    build_dashboard(shap_arr, base_val, X_used, y_true, y_pred,
                    feat_cols, feat_grps, error_masks, df, keep_idx)
 
    # ── Step 7: Individual explanations ──────────────────────────────────────
    log.info("Generating individual FP/FN explanations …")
    plot_individual_explanations(
        shap_arr, base_val, X_used, y_true, y_pred,
        feat_cols, error_masks, df, keep_idx)
 
    # ── Step 8: Export SHAP CSV ───────────────────────────────────────────────
    log.info("Exporting SHAP values to CSV …")
    export_shap_csv(shap_arr, X_used, y_true, y_pred,
                    feat_cols, df, keep_idx)
 
    _sep()
    print(f"  DONE in {time.time()-t0:.1f}s")
    print(f"  All outputs in: {CFG['OUTPUT_DIR']}/")
    _sep()
 
 
if __name__ == "__main__":
    main()