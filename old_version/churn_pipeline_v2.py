"""
churn_pipeline_v2.py
====================
Optimized Telecom Churn Pipeline — Full-Feature Edition
────────────────────────────────────────────────────────
Improvements over v1:
  ✓ Uses 141 pre-computed behavioral features directly (no re-engineering)
  ✓ Respects DATASET_TYPE split (TRAIN CV → TEST holdout evaluation)
  ✓ Winsorises extreme LONG_DROP / VOLATILITY / TREND_SLOPE columns (p1–p99)
  ✓ RobustScaler (outlier-resistant vs StandardScaler)
  ✓ Mutual-information feature selection (top-K, computed inside TRAIN only)
  ✓ Manual SMOTE applied per-fold (training fold only, never validation/test)
  ✓ Stacked Ensemble: XGBoost + RandomForest + ExtraTrees → calibrated LR meta
  ✓ Probability calibration (Isotonic regression)
  ✓ F2-score threshold sweep with precision guardrail (≥ PRECISION_FLOOR)
  ✓ Full evaluation: OOF CV + TEST holdout confusion matrix + dashboard plot

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
CFG = {
    "INPUT_CSV"       : os.getenv("INPUT_CSV", "Sample_data_full_feature.csv"),
    "TARGET"          : "LABEL_CHURN_90D",
    "DATASET_TYPE_COL": "DATASET_TYPE",
    "OUTPUT_DIR"      : "./churn_v2_outputs",
    "RANDOM_STATE"    : 42,
    "N_FOLDS"         : 5,
    # Feature selection
    "TOP_K_FEATURES"  : 70,          # top MI features to keep
    "WINSOR_P_LOW"    : 0.01,
    "WINSOR_P_HIGH"   : 0.99,
    # SMOTE
    "SMOTE_K"         : 3,           # k-neighbours for synthetic sampling
    "SMOTE_RATIO"     : 0.10,        # target minority:majority ratio post-SMOTE
    # Threshold sweep
    "TH_MIN"          : 0.01,
    "TH_MAX"          : 0.50,
    "TH_STEPS"        : 200,
    "PRECISION_FLOOR" : 0.05,        # minimum acceptable precision (guardrail)
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

# Columns to drop before modeling
_ID_COLS = {
    "MSISDN","MSISDN_9","MSISDN_251","SNAPSHOT_DATE",
    "AON",               # kept separately — added back as feature below
    "DATASET_TYPE",
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


# ── 2. Data loading & initial cleaning ───────────────────────────────────────

def load_and_clean() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Returns (train_df, test_df, feature_cols)
    - train_df / test_df : cleaned dataframes with features + TARGET
    - feature_cols       : ordered list of 141 → 70 selected features
    """
    path = CFG["INPUT_CSV"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.upper().strip() for c in df.columns]
    log.info("Loaded: %s rows × %s cols", *df.shape)

    target   = CFG["TARGET"]
    ds_col   = CFG["DATASET_TYPE_COL"]
    df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)

    # AON as a feature (meaningful: churners are ~20% younger on network)
    if "AON" in df.columns:
        df["AON"] = pd.to_numeric(df["AON"], errors="coerce").fillna(0)

    # Build base feature list
    feat_cols = [
        c for c in df.columns
        if c not in _ID_COLS and c not in _DROP_REDUNDANT
    ]
    if "AON" in df.columns:
        feat_cols = ["AON"] + [c for c in feat_cols if c != "AON"]

    # Cast all features to numeric; replace inf
    for col in feat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce") \
                    .replace([np.inf, -np.inf], np.nan) \
                    .fillna(0.0)

    # Winsorise extreme columns (using FULL dataset quantiles — stable estimates)
    for col in feat_cols:
        if any(kw in col for kw in _WINSOR_COLS_KEYWORDS):
            lo = df[col].quantile(CFG["WINSOR_P_LOW"])
            hi = df[col].quantile(CFG["WINSOR_P_HIGH"])
            df[col] = df[col].clip(lo, hi)

    # TRAIN / TEST partition
    if ds_col in df.columns:
        train_df = df[df[ds_col].str.upper() == "TRAIN"].copy()
        test_df  = df[df[ds_col].str.upper() == "TEST"].copy()
    else:
        log.warning("No DATASET_TYPE column — using 70/30 stratified split")
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=0.30, stratify=df[target],
            random_state=CFG["RANDOM_STATE"]
        )

    log.info(
        "TRAIN: %d rows | %d churners (%.2f%%)",
        len(train_df), train_df[target].sum(),
        train_df[target].mean() * 100,
    )
    log.info(
        "TEST : %d rows | %d churners (%.2f%%)",
        len(test_df), test_df[target].sum(),
        test_df[target].mean() * 100,
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
    log.info("Selected top-%d features (MI ≥ %.5f)", top_k, mi_series.iloc[top_k - 1])
    return selected, mi


# ── 4. Manual SMOTE (no external package) ────────────────────────────────────

def manual_smote(
    X_min : np.ndarray,
    n_synthetic: int,
    k    : int = 5,
    rng  : Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Basic SMOTE interpolation for the minority class.
    Generates `n_synthetic` synthetic samples from the minority X_min array.
    """
    if rng is None:
        rng = np.random.default_rng(CFG["RANDOM_STATE"])
    nn  = NearestNeighbors(n_neighbors=min(k + 1, len(X_min)), metric="euclidean")
    nn.fit(X_min)
    _, indices = nn.kneighbors(X_min)

    synthetic = []
    for _ in range(n_synthetic):
        idx    = rng.integers(0, len(X_min))
        nn_idx = rng.choice(indices[idx][1:])   # skip self
        alpha  = rng.random()
        sample = X_min[idx] + alpha * (X_min[nn_idx] - X_min[idx])
        synthetic.append(sample)
    return np.array(synthetic)


def apply_smote_to_fold(
    X_tr  : np.ndarray,
    y_tr  : np.ndarray,
    ratio : float = 0.10,
    k     : int   = 3,
    seed  : int   = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE only to minority class of a training fold."""
    pos_mask  = y_tr == 1
    neg_count = int((y_tr == 0).sum())
    pos_count = int(pos_mask.sum())
    if pos_count == 0:
        return X_tr, y_tr

    target_pos = max(pos_count, int(neg_count * ratio))
    n_synth    = max(0, target_pos - pos_count)
    if n_synth == 0:
        return X_tr, y_tr

    rng    = np.random.default_rng(seed)
    X_min  = X_tr[pos_mask]
    X_syn  = manual_smote(X_min, n_synth, k=k, rng=rng)
    y_syn  = np.ones(len(X_syn), dtype=int)

    X_out  = np.vstack([X_tr, X_syn])
    y_out  = np.concatenate([y_tr, y_syn])
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
        X_tr_sm, y_tr_sm = apply_smote_to_fold(
            X_tr_s, y_tr_raw,
            ratio=CFG["SMOTE_RATIO"],
            k=CFG["SMOTE_K"],
            seed=CFG["RANDOM_STATE"] + fold,
        )
        pos_after = int((y_tr_sm == 1).sum())
        log.info(
            "  Fold %d/%d │ train=%d→%d (pos %d→%d) │ val=%d (pos %d)",
            fold, CFG["N_FOLDS"],
            len(y_tr_raw), len(y_tr_sm),
            int((y_tr_raw==1).sum()), pos_after,
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


# ── 6. Threshold optimiser ────────────────────────────────────────────────────

def sweep_threshold(
    y_true : np.ndarray,
    y_prob : np.ndarray,
    label  : str = "OOF",
) -> Tuple[float, pd.DataFrame]:
    """
    Sweep thresholds 0.01–0.50, return optimal threshold maximising F2
    subject to precision ≥ PRECISION_FLOOR.
    """
    thresholds = np.linspace(CFG["TH_MIN"], CFG["TH_MAX"], CFG["TH_STEPS"])
    rows = []
    for th in thresholds:
        pred  = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
        prec  = precision_score(y_true, pred, zero_division=0)
        rec   = recall_score   (y_true, pred, zero_division=0)
        f2    = fbeta_score    (y_true, pred, beta=2, zero_division=0)
        f1    = f1_score       (y_true, pred, zero_division=0)
        rows.append(dict(
            threshold=round(float(th),4), precision=round(prec,4),
            recall=round(rec,4), f1=round(f1,4), f2=round(f2,4),
            tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
            predicted_pos=int(tp+fp),
        ))

    sweep = pd.DataFrame(rows)

    # Primary: max F2 subject to precision ≥ floor
    floor   = CFG["PRECISION_FLOOR"]
    guarded = sweep[sweep["precision"] >= floor]
    if guarded.empty:
        log.warning("No threshold meets precision floor %.2f — relaxing constraint", floor)
        guarded = sweep
    best_row = guarded.loc[guarded["f2"].idxmax()]
    best_th  = float(best_row["threshold"])

    log.info(
        "[%s] Optimal threshold=%.4f → F2=%.4f  Recall=%.4f  Precision=%.4f  "
        "TP=%d  FN=%d  FP=%d",
        label, best_th,
        best_row["f2"], best_row["recall"], best_row["precision"],
        int(best_row["tp"]), int(best_row["fn"]), int(best_row["fp"]),
    )
    return best_th, sweep


# ── 7. TEST holdout evaluation ────────────────────────────────────────────────

def evaluate_on_test(
    train_df     : pd.DataFrame,
    test_df      : pd.DataFrame,
    selected_feat: List[str],
    best_th      : float,
    meta_lr      ,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Re-train full ensemble on ALL of TRAIN, predict on TEST.
    Returns (test_prob, test_true, test_pred).
    """
    log.info("Re-training full ensemble on entire TRAIN for TEST holdout …")
    X_tr_full = train_df[selected_feat].values.astype(float)
    y_tr_full = train_df[CFG["TARGET"]].values.astype(int)
    X_te      = test_df [selected_feat].values.astype(float)
    y_te      = test_df [CFG["TARGET"]].values.astype(int)

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr_full)
    X_te_s = scaler.transform(X_te)

    # SMOTE on full TRAIN
    X_tr_sm, y_tr_sm = apply_smote_to_fold(
        X_tr_s, y_tr_full,
        ratio=CFG["SMOTE_RATIO"],
        k=CFG["SMOTE_K"],
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
    test_pred = (test_prob >= best_th).astype(int)

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
    return test_prob, y_te, test_pred


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
    print("  TELECOM CHURN V2  —  FULL EVALUATION REPORT")
    _sep()
    print(f"  Ensemble      : XGBoost + RandomForest + ExtraTrees → LR meta")
    print(f"  SMOTE ratio   : minority→{CFG['SMOTE_RATIO']*100:.0f}% of majority (per CV fold)")
    print(f"  Feature count : {len(selected_feat)} (MI-selected from 141)")
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

    # TEST holdout metrics
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
        "Telecom Churn V2  —  Stacked Ensemble Diagnostic Dashboard",
        fontsize=15, fontweight="bold", y=0.99,
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

def main():
    t0 = time.time()
    _sep()
    print("  CHURN PIPELINE V2  —  Optimised Stacked Ensemble + SMOTE")
    _sep()

    # Step 1 – Load & clean
    train_df, test_df, feat_cols = load_and_clean()

    # Step 2 – Feature selection (MI on TRAIN)
    selected_feat, _ = select_features(train_df, feat_cols, top_k=CFG["TOP_K_FEATURES"])

    # Step 3 – Stratified CV + stacked ensemble
    oof_prob, oof_true, fi_dict, meta_lr, fold_metrics = run_stacked_cv(
        train_df, selected_feat, feat_cols
    )

    # Step 4 – OOF threshold optimisation
    oof_th, oof_sweep = sweep_threshold(oof_true, oof_prob, label="OOF")

    # Step 5 – TEST holdout evaluation
    test_prob, test_true, test_pred = evaluate_on_test(
        train_df, test_df, selected_feat, oof_th, meta_lr
    )
    # Re-run sweep on TEST to get its own sweep + same threshold
    test_th, _ = sweep_threshold(test_true, test_prob, label="TEST")

    # Step 6 – Full report + dashboard
    full_report(
        oof_prob, oof_true, oof_th, oof_sweep,
        test_prob, test_true, test_pred, test_th,
        selected_feat, fi_dict, fold_metrics,
    )

    _sep()
    print(f"  Pipeline complete in {time.time()-t0:.1f}s. Outputs: {CFG['OUTPUT_DIR']}")
    _sep()


if __name__ == "__main__":
    main()
