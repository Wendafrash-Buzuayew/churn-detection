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
from sklearn.model_selection import StratifiedKFold, train_test_split

from churn_ranker import evaluation, features, schema, tiers

_LOGIT_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    """Log-odds transform, clipped away from 0/1 so scores near the extremes -- exactly
    where tier thresholds live -- don't blow up to +/-inf."""
    clipped = np.clip(np.asarray(p, dtype=float), _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(clipped / (1.0 - clipped))


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
    use_monotonic_constraints: bool = False
    # Calibration + tier thresholds are set from a genuine held-out slice, scored by the
    # deployed model itself, whenever there's enough of the minority class to trust it --
    # otherwise this falls back to the OOF ensemble (see ChurnRanker.fit).
    calibration_holdout_frac: float = 0.15
    min_calibration_positives: int = 20
    min_calibration_negatives: int = 20


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
        if fit:
            base = canonical[schema.feature_columns(canonical)]
            full = pd.concat([base, derived], axis=1)
            self.feature_names_ = list(full.columns)
        else:
            # Build the base frame from the trained base feature names rather than
            # re-deriving them from this chunk's dtypes: a chunk where a numeric
            # column parses as object must still be coerced, not silently dropped
            # (which would leave the trained feature NaN after the reindex below).
            base_names = [n for n in self.feature_names_ if not n.startswith("FE_")]
            base = pd.DataFrame(
                {
                    c: pd.to_numeric(canonical[c], errors="coerce")
                    if c in canonical.columns
                    else np.full(len(canonical), np.nan)
                    for c in base_names
                },
                index=canonical.index,
            )
            full = pd.concat([base, derived], axis=1)
        full = full.reindex(columns=self.feature_names_)
        return full.to_numpy(dtype=np.float32)

    def _base_model(self) -> HistGradientBoostingClassifier:
        c = self.config
        kwargs = dict(
            max_iter=c.max_iter,
            learning_rate=c.learning_rate,
            max_leaf_nodes=c.max_leaf_nodes,
            min_samples_leaf=c.min_samples_leaf,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=c.random_state,
        )
        if c.use_monotonic_constraints and self.feature_names_:
            kwargs["monotonic_cst"] = features.monotonic_constraints(self.feature_names_)
        return HistGradientBoostingClassifier(**kwargs)

    def _can_use_calibration_holdout(self, positives: int, negatives: int) -> bool:
        c = self.config
        if not (0.0 < c.calibration_holdout_frac < 1.0):
            return False
        expected_holdout_positives = positives * c.calibration_holdout_frac
        expected_holdout_negatives = negatives * c.calibration_holdout_frac
        return (
            expected_holdout_positives >= c.min_calibration_positives
            and expected_holdout_negatives >= c.min_calibration_negatives
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
        n = len(y)
        positives, negatives = int(y.sum()), n - int(y.sum())
        if min(self.config.cv_folds, positives, negatives) < 2:
            raise ValueError("Not enough minority examples for cross-validation")

        use_holdout = self._can_use_calibration_holdout(positives, negatives)
        if use_holdout:
            all_idx = np.arange(n)
            fit_idx, holdout_idx = train_test_split(
                all_idx,
                test_size=self.config.calibration_holdout_frac,
                stratify=y,
                random_state=self.config.random_state,
            )
        else:
            fit_idx, holdout_idx = np.arange(n), None

        X_fit, y_fit = X[fit_idx], y[fit_idx]
        fit_folds = min(self.config.cv_folds, int(y_fit.sum()), int(len(y_fit) - y_fit.sum()))
        if fit_folds < 2:
            raise ValueError("Not enough minority examples for cross-validation")

        oof = np.zeros(len(y_fit))
        splitter = StratifiedKFold(fit_folds, shuffle=True, random_state=self.config.random_state)
        for train_idx, valid_idx in splitter.split(X_fit, y_fit):
            fold_model = self._base_model()
            fold_model.fit(X_fit[train_idx], y_fit[train_idx])
            oof[valid_idx] = fold_model.predict_proba(X_fit[valid_idx])[:, 1]

        self.model = self._base_model()
        self.model.fit(X_fit, y_fit)

        if holdout_idx is not None:
            calibration_source = "held_out"
            calib_raw = self.model.predict_proba(X[holdout_idx])[:, 1]
            calib_y = y[holdout_idx]
        else:
            calibration_source = "oof"
            calib_raw = oof
            calib_y = y_fit

        self.calibrator = LogisticRegression(random_state=self.config.random_state)
        calib_logit = _logit(calib_raw)
        self.calibrator.fit(calib_logit.reshape(-1, 1), calib_y)
        calibrated = self.calibrator.predict_proba(calib_logit.reshape(-1, 1))[:, 1]
        self.tier_thresholds_ = tiers.tier_thresholds(calibrated, self.config.tier_spec)

        self.training_summary_ = {
            "n_rows": int(n),
            "n_churners": int(y.sum()),
            "n_features": len(self.feature_names_),
            "cv_folds": fit_folds,
            "tier_thresholds": [[name, float(t)] for name, t in self.tier_thresholds_],
            "oof_metrics": evaluation.ranking_metrics(y_fit, oof),
            "oof_lift_table": evaluation.lift_table(y_fit, oof).to_dict(orient="records"),
            "calibration": {
                "source": calibration_source,
                "n": int(len(calib_y)),
                "brier_score": evaluation.brier_score(calib_y, calibrated),
                "ece": evaluation.expected_calibration_error(calib_y, calibrated),
            },
        }
        return self.training_summary_

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.calibrator is None:
            raise RuntimeError("ChurnRanker is not fitted")
        raw = self.model.predict_proba(self._matrix(df))[:, 1]
        return self.calibrator.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]

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
