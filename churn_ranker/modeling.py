"""Calibrated churn risk ranker with training-time capacity-tier thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from churn_ranker import evaluation, features, schema, tiers


@dataclass
class RankerConfig:
    target: str = schema.TARGET
    cv_folds: int = 5
    random_state: int = 42
    max_iter: int = 300
    learning_rate: float = 0.06
    max_leaf_nodes: int = 63
    min_samples_leaf: int = 60
    tier_spec: tuple = tiers.DEFAULT_TIERS


class ChurnRanker:
    def __init__(self, config: RankerConfig | None = None):
        self.config = config or RankerConfig()
        self.model: HistGradientBoostingClassifier | None = None
        self.calibrator: LogisticRegression | None = None
        self.feature_names_: list[str] = []
        self.tier_thresholds_: list[tuple[str, float]] = []
        self.training_summary_: dict[str, Any] = {}

    def _matrix(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        canonical = schema.to_canonical(df)
        derived = features.build_features(canonical)
        base = canonical[schema.feature_columns(canonical)]
        full = pd.concat([base, derived], axis=1)
        if fit:
            self.feature_names_ = list(full.columns)
        full = full.reindex(columns=self.feature_names_)
        return full.to_numpy(dtype=np.float32)

    def _base_model(self) -> HistGradientBoostingClassifier:
        c = self.config
        return HistGradientBoostingClassifier(
            max_iter=c.max_iter,
            learning_rate=c.learning_rate,
            max_leaf_nodes=c.max_leaf_nodes,
            min_samples_leaf=c.min_samples_leaf,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=c.random_state,
        )

    def fit(self, df: pd.DataFrame) -> dict[str, Any]:
        canonical = schema.to_canonical(df)
        if self.config.target not in canonical.columns:
            raise ValueError(f"Training data must contain {self.config.target}")
        y = (
            pd.to_numeric(canonical[self.config.target], errors="coerce")
            .fillna(0).astype(int).to_numpy()
        )
        if len(np.unique(y)) < 2:
            raise ValueError("Training data needs both churn classes")
        X = self._matrix(df, fit=True)
        folds = min(self.config.cv_folds, int(y.sum()), int(len(y) - y.sum()))
        if folds < 2:
            raise ValueError("Not enough minority examples for cross-validation")
        oof = np.zeros(len(y))
        splitter = StratifiedKFold(folds, shuffle=True, random_state=self.config.random_state)
        for train_idx, valid_idx in splitter.split(X, y):
            fold_model = self._base_model()
            fold_model.fit(X[train_idx], y[train_idx])
            oof[valid_idx] = fold_model.predict_proba(X[valid_idx])[:, 1]
        self.calibrator = LogisticRegression(random_state=self.config.random_state)
        self.calibrator.fit(oof.reshape(-1, 1), y)
        calibrated = self.calibrator.predict_proba(oof.reshape(-1, 1))[:, 1]
        self.tier_thresholds_ = tiers.tier_thresholds(calibrated, self.config.tier_spec)
        self.model = self._base_model()
        self.model.fit(X, y)
        self.training_summary_ = {
            "n_rows": int(len(y)),
            "n_churners": int(y.sum()),
            "n_features": len(self.feature_names_),
            "cv_folds": folds,
            "tier_thresholds": [[name, float(t)] for name, t in self.tier_thresholds_],
            "oof_metrics": evaluation.ranking_metrics(y, calibrated),
            "oof_lift_table": evaluation.lift_table(y, calibrated).to_dict(orient="records"),
        }
        return self.training_summary_

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.calibrator is None:
            raise RuntimeError("ChurnRanker is not fitted")
        raw = self.model.predict_proba(self._matrix(df))[:, 1]
        return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        probability = self.predict_proba(df)
        derived = features.build_features(schema.to_canonical(df))
        result = df.copy()
        result["churn_probability"] = probability
        tier_labels = tiers.assign_tiers(probability, self.tier_thresholds_)
        result["risk_tier"] = tier_labels
        result["reason_code"] = tiers.reason_codes(derived, tier_labels)
        return result

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ChurnRanker":
        return joblib.load(path)
