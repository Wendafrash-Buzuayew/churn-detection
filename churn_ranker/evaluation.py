"""Ranking-quality metrics: AUCs and capacity lift tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TOP_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20)


def ranking_metrics(y_true, scores) -> dict:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    out = {"n": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean())}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def lift_table(y_true, scores, top_fractions=TOP_FRACTIONS) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) == 0:
        return pd.DataFrame(columns=[
            "top_fraction", "contacted", "churners_caught", "precision", "recall", "lift",
        ])
    order = np.argsort(-s, kind="stable")
    cumulative_positives = np.cumsum(y[order])
    total_positives = max(int(y.sum()), 1)
    base_rate = max(float(y.mean()), 1e-12)
    rows = []
    for fraction in top_fractions:
        contacted = max(1, int(round(len(y) * fraction)))
        caught = int(cumulative_positives[contacted - 1])
        precision = caught / contacted
        rows.append({
            "top_fraction": fraction,
            "contacted": contacted,
            "churners_caught": caught,
            "precision": precision,
            "recall": caught / total_positives,
            "lift": precision / base_rate,
        })
    return pd.DataFrame(rows)


def brier_score(y_true, scores) -> float:
    """Mean squared error between calibrated probability and outcome (lower is better)."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    if len(y) == 0:
        return 0.0
    return float(np.mean((s - y) ** 2))


def expected_calibration_error(y_true, scores, n_bins: int = 10) -> float:
    """Weighted mean gap between predicted probability and observed rate, per score bin."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    n = len(y)
    if n == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(s, bin_edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        bin_confidence = s[mask].mean()
        bin_accuracy = y[mask].mean()
        ece += (mask.sum() / n) * abs(bin_confidence - bin_accuracy)
    return float(ece)
