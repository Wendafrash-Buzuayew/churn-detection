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
