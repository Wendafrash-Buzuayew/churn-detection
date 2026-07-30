"""Production-oriented hybrid engine for 30-day telecom churn prediction."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

LOG = logging.getLogger("unified_churn_engine")
WEEKS = (10, 11, 12, 13)
SERVICE_COLUMNS = {
    "DATA": ("DATA_MB", "DATA_VOLUME_MB"),
    "VOICE": ("OG_VOICE_MIN", "TOTAL_VOICE_MIN", "VOICE_MIN"),
    "BUNDLE": ("BUNDLE_CNT", "BUNDLE_COUNT"),
    "REVENUE": ("TOTAL_REVENUE", "DATA_REVENUE", "BUNDLE_REVENUE"),
}


@dataclass
class EngineConfig:
    target: str = "LABEL_CHURN_30D"
    dataset_type_col: str = "DATASET_TYPE"
    precision_floor: float = 0.70
    high_confidence_probability: float = 0.90
    cv_folds: int = 5
    random_state: int = 42
    test_size: float = 0.30
    clip_low: float = 0.001
    clip_high: float = 0.999
    max_iter: int = 250
    learning_rate: float = 0.06
    max_leaf_nodes: int = 31


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).upper().strip() for c in out.columns]
    return out


def resolve_target(df: pd.DataFrame, configured_target: str = "LABEL_CHURN_30D") -> Optional[str]:
    columns = {str(c).upper().strip(): c for c in df.columns}
    if configured_target.upper() in columns:
        return str(columns[configured_target.upper()]).upper().strip()
    return None


def _has_two_classes(y: pd.Series) -> bool:
    return y.nunique(dropna=True) == 2 and y.value_counts().min() >= 2


def resolve_split(df: pd.DataFrame, target: Optional[str], config: EngineConfig) -> tuple[pd.DataFrame, Optional[pd.DataFrame], str]:
    """Return training/evaluation frames; unlabeled or OOT-only input is score-only."""
    if target is None or target not in df or not _has_two_classes(pd.to_numeric(df[target], errors="coerce").fillna(0)):
        return df, None, "score_only"
    labels = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)
    tag = config.dataset_type_col
    if tag in df:
        groups = df[tag].fillna("").astype(str).str.upper().str.strip()
        train = df.loc[groups.eq("TRAIN")]
        test = df.loc[groups.eq("TEST")]
        if len(train) and len(test) and _has_two_classes(train[target]):
            return train.reset_index(drop=True), test.reset_index(drop=True), "explicit_train_test"
        oot = df.loc[groups.eq("OOT")]
        if len(train) and len(oot) and _has_two_classes(train[target]):
            return train.reset_index(drop=True), oot.reset_index(drop=True), "train_oot"
        if groups.eq("OOT").all():
            return df, None, "score_only_oot"
    train, test = train_test_split(df, test_size=config.test_size, stratify=labels,
                                   random_state=config.random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True), "stratified_fallback"


class HybridFeatureBuilder:
    """Vectorized telecom rule/meta-feature generator; no row-wise apply."""
    def __init__(self) -> None:
        self.stable_low_frequency_threshold_: Optional[float] = None

    def fit(self, df: pd.DataFrame) -> "HybridFeatureBuilder":
        raw = normalize_columns(df)
        matrices = {name: self._matrix(raw, prefixes) for name, prefixes in SERVICE_COLUMNS.items()}
        core = ("DATA", "VOICE", "BUNDLE")
        baseline_activity = sum(matrices[n][:, :3].mean(axis=1) for n in core)
        self.stable_low_frequency_threshold_ = float(np.nanmedian(baseline_activity) + 1)
        return self

    def _matrix(self, df: pd.DataFrame, candidates: tuple[str, ...]) -> np.ndarray:
        prefix = next((p for p in candidates if any(f"{p}_W{w}" in df for w in WEEKS)), None)
        if prefix is None:
            return np.zeros((len(df), 4), dtype=np.float32)
        values = [pd.to_numeric(df.get(f"{prefix}_W{w}", 0.0), errors="coerce").fillna(0).to_numpy(dtype=np.float32)
                  if isinstance(df.get(f"{prefix}_W{w}", 0.0), pd.Series)
                  else np.full(len(df), float(df.get(f"{prefix}_W{w}", 0.0)), dtype=np.float32)
                  for w in WEEKS]
        return np.column_stack(values).astype(np.float32, copy=False)

    @staticmethod
    def _terminal_run(matrix: np.ndarray) -> np.ndarray:
        zeros = matrix <= 0
        run = np.zeros(len(matrix), dtype=np.int8)
        active_seen = np.zeros(len(matrix), dtype=bool)
        for position in range(3, -1, -1):
            run = np.where(~active_seen & zeros[:, position], run + 1, run).astype(np.int8)
            active_seen |= ~zeros[:, position]
        return run

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raw = normalize_columns(df)
        out = pd.DataFrame(index=raw.index)
        matrices = {name: self._matrix(raw, prefixes) for name, prefixes in SERVICE_COLUMNS.items()}
        core = ("DATA", "VOICE", "BUNDLE")
        collapse_flags, terminal_runs = [], []
        for name in core:
            matrix = matrices[name]
            baseline = matrix[:, :3].mean(axis=1)
            ratio = np.divide(matrix[:, 3], baseline, out=np.zeros(len(raw), dtype=np.float32), where=baseline > 0)
            drop = np.maximum(0, 1 - ratio)
            slope = (matrix[:, 3] - matrix[:, 0]) / (np.abs(matrix[:, 0]) + 1.0)
            weighted = matrix @ np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
            run = self._terminal_run(matrix)
            collapse = (baseline > 0) & (ratio <= 0.20)
            out[f"{name}_W13_BASELINE_RATIO"] = ratio
            out[f"{name}_W13_DROP_PCT"] = drop
            out[f"{name}_RECENT_DECAY_SLOPE"] = slope
            out[f"{name}_RECENCY_WEIGHTED_ACTIVITY"] = weighted
            out[f"{name}_TERMINAL_ZERO_RUN"] = run
            out[f"RULE_{name}_W13_COLLAPSE"] = collapse.astype(np.int8)
            collapse_flags.append(collapse)
            terminal_runs.append(run)
        collapse_count = np.column_stack(collapse_flags).sum(axis=1)
        zero_breadth = np.column_stack([matrices[n][:, 3] <= 0 for n in core]).sum(axis=1)
        terminal_multi = zero_breadth >= 2
        ratio_stack = out[[f"{n}_W13_BASELINE_RATIO" for n in core]].to_numpy(dtype=np.float32)
        out["RULE_W13_COLLAPSE_BREADTH"] = collapse_count.astype(np.int8)
        out["RULE_W13_ZERO_BREADTH"] = zero_breadth.astype(np.int8)
        out["RULE_TERMINAL_MULTI_SERVICE"] = terminal_multi.astype(np.int8)
        out["RULE_ALL_SERVICES_ZERO_W13"] = (zero_breadth == 3).astype(np.int8)
        out["MAX_TERMINAL_ZERO_RUN"] = np.maximum.reduce(terminal_runs)
        out["CROSS_SERVICE_COLLAPSE_DIVERGENCE"] = ratio_stack.max(axis=1) - ratio_stack.min(axis=1)
        out["RULE_RECOVERING_MULTI_SERVICE"] = np.column_stack([matrices[n][:, 3] > matrices[n][:, 0] for n in core]).sum(axis=1).astype(np.int8)
        revenue = matrices["REVENUE"]
        rev_base = revenue[:, :3].mean(axis=1)
        rev_ratio = np.divide(revenue[:, 3], rev_base, out=np.zeros(len(raw), dtype=np.float32), where=rev_base > 0)
        out["REVENUE_W13_BASELINE_RATIO"] = rev_ratio
        out["RULE_RECHARGE_CLIFF"] = ((rev_base > 0) & (rev_ratio <= .15)).astype(np.int8)
        aon = pd.to_numeric(raw.get("AON", 0), errors="coerce")
        if not isinstance(aon, pd.Series): aon = pd.Series(aon, index=raw.index)
        out["AON_LOG"] = np.log1p(aon.fillna(0).clip(lower=0)).astype(np.float32)
        out["RULE_SHORT_TENURE"] = (aon.fillna(0) < 90).astype(np.int8)
        baseline_activity = sum(matrices[n][:, :3].mean(axis=1) for n in core)
        active_late = (matrices["DATA"][:, 3] > 0) | (matrices["VOICE"][:, 3] > 0) | (revenue[:, 3] > 0)
        if self.stable_low_frequency_threshold_ is None:
            self.stable_low_frequency_threshold_ = float(np.nanmedian(baseline_activity) + 1)
        out["GUARD_STABLE_LOW_FREQUENCY_PAYER"] = ((baseline_activity > 0) &
                                                     (baseline_activity < self.stable_low_frequency_threshold_) &
                                                     active_late & ~terminal_multi).astype(np.int8)
        early = sum(matrices[n][:, :2].sum(axis=1) for n in core)
        late = sum(matrices[n][:, 2:].sum(axis=1) for n in core)
        out["GUARD_CYCLICAL_RECHARGER"] = ((late > (early * 2 + 1)) & (revenue[:, 3] > 0) & ~terminal_multi).astype(np.int8)
        out["RULE_SCORE"] = (2.5 * collapse_count + 2.0 * terminal_multi + 1.5 * out["RULE_RECHARGE_CLIFF"]
                             - 1.5 * out["RULE_RECOVERING_MULTI_SERVICE"] - out["GUARD_STABLE_LOW_FREQUENCY_PAYER"]).astype(np.float32)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


class UnifiedChurnEngine:
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.builder = HybridFeatureBuilder()
        self.model: Optional[HistGradientBoostingClassifier] = None
        self.calibrator: Optional[LogisticRegression] = None
        self.feature_columns_: list[str] = []
        self.medians_: Optional[pd.Series] = None
        self.clip_bounds_: Optional[pd.DataFrame] = None
        self.threshold_: float = 0.5
        self.training_summary_: dict[str, Any] = {}

    def _prepare_fit(self, df: pd.DataFrame) -> np.ndarray:
        self.builder.fit(df)
        features = self.builder.transform(df)
        self.feature_columns_ = list(features.columns)
        self.medians_ = features.median().fillna(0.0)
        self.clip_bounds_ = pd.DataFrame({"lo": features.quantile(self.config.clip_low), "hi": features.quantile(self.config.clip_high)})
        return self._prepare_transform(df)

    def _prepare_transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None or self.clip_bounds_ is None: raise RuntimeError("Engine is not fitted")
        features = self.builder.transform(df).reindex(columns=self.feature_columns_, fill_value=0.0)
        features = features.fillna(self.medians_).clip(self.clip_bounds_["lo"], self.clip_bounds_["hi"], axis=1)
        return features.to_numpy(dtype=np.float32)

    @staticmethod
    def _metrics(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
        result: dict[str, Any] = {"precision": float(precision_score(y, pred, zero_division=0)),
                                   "recall": float(recall_score(y, pred, zero_division=0)),
                                   "f1": float(f1_score(y, pred, zero_division=0)),
                                   "flagged": int(pred.sum()), "population": int(len(y)),
                                   "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}
        if len(np.unique(y)) == 2:
            result["roc_auc"] = float(roc_auc_score(y, p)); result["pr_auc"] = float(average_precision_score(y, p))
        else: result["roc_auc"] = None; result["pr_auc"] = None
        return result

    @staticmethod
    def _threshold(y: np.ndarray, p: np.ndarray, floor: float) -> float:
        precision, recall, thresholds = precision_recall_curve(y, p)
        if not len(thresholds): return 0.5
        viable = np.flatnonzero(precision[:-1] >= floor)
        idx = viable[np.argmax(recall[:-1][viable])] if viable.size else int(np.argmax(precision[:-1]))
        return float(thresholds[idx])

    def _base_model(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(max_iter=self.config.max_iter, learning_rate=self.config.learning_rate,
                                              max_leaf_nodes=self.config.max_leaf_nodes, l2_regularization=1.0,
                                              random_state=self.config.random_state)

    def fit(self, df: pd.DataFrame) -> dict[str, Any]:
        raw = normalize_columns(df); target = resolve_target(raw, self.config.target)
        train, test, strategy = resolve_split(raw, target, self.config)
        if strategy.startswith("score_only") or target is None: raise ValueError("Training requires labeled data with both churn classes")
        y = pd.to_numeric(train[target], errors="coerce").fillna(0).astype(int).to_numpy()
        X = self._prepare_fit(train)
        minority = int(y.sum()); folds = min(self.config.cv_folds, minority, len(y) - minority)
        if folds < 2: raise ValueError("Training needs at least two positive and two negative records")
        oof = np.zeros(len(y), dtype=np.float64)
        for tr, va in StratifiedKFold(folds, shuffle=True, random_state=self.config.random_state).split(X, y):
            model = self._base_model(); yt = y[tr]; weights = np.where(yt == 1, (len(yt)-yt.sum()) / max(yt.sum(), 1), 1.0)
            model.fit(X[tr], yt, sample_weight=weights); oof[va] = model.predict_proba(X[va])[:, 1]
        calibrator_weights = np.where(y == 1, (len(y) - y.sum()) / max(y.sum(), 1), 1.0)
        self.calibrator = LogisticRegression(random_state=self.config.random_state).fit(oof.reshape(-1, 1), y, sample_weight=calibrator_weights)
        calibrated_oof = self.calibrator.predict_proba(oof.reshape(-1, 1))[:, 1]
        self.threshold_ = self._threshold(y, calibrated_oof, self.config.precision_floor)
        self.model = self._base_model(); weights = np.where(y == 1, (len(y)-y.sum()) / max(y.sum(), 1), 1.0); self.model.fit(X, y, sample_weight=weights)
        self.training_summary_ = {"split_strategy": strategy, "target": target, "threshold": self.threshold_,
                                  "oof": self._metrics(y, calibrated_oof, calibrated_oof >= self.threshold_)}
        if test is not None and target in test and _has_two_classes(pd.to_numeric(test[target], errors="coerce").fillna(0)):
            scored = self.score(test); yt = pd.to_numeric(test[target], errors="coerce").fillna(0).astype(int).to_numpy()
            self.training_summary_["evaluation"] = self._metrics(yt, scored["churn_probability"].to_numpy(), scored["churn_prediction"].to_numpy())
            self.training_summary_["guard_impact"] = {"suppressed": int(scored["false_positive_suppressed"].sum()),
                "suppressed_churn_rate": float(test.loc[scored["false_positive_suppressed"], target].mean()) if scored["false_positive_suppressed"].any() else None}
        return self.training_summary_

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.calibrator is None: raise RuntimeError("Engine is not fitted")
        raw = self.model.predict_proba(self._prepare_transform(normalize_columns(df)))[:, 1]
        return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = normalize_columns(df); signals = self.builder.transform(normalized); probability = self.predict_proba(normalized)
        candidate = probability >= self.threshold_
        terminal = signals["RULE_TERMINAL_MULTI_SERVICE"].to_numpy(dtype=bool)
        guards = (signals["GUARD_STABLE_LOW_FREQUENCY_PAYER"].to_numpy(dtype=bool) |
                  signals["GUARD_CYCLICAL_RECHARGER"].to_numpy(dtype=bool))
        borderline = probability < self.config.high_confidence_probability
        suppressed = candidate & borderline & guards & ~terminal
        prediction = candidate & ~suppressed
        result = df.copy(); result["churn_probability"] = probability; result["rule_score"] = signals["RULE_SCORE"].to_numpy()
        result["false_positive_suppressed"] = suppressed; result["churn_prediction"] = prediction.astype(np.int8)
        result["risk_tier"] = np.select([prediction & (probability >= self.config.high_confidence_probability), prediction,
                                         probability >= max(self.threshold_ * .75, .01)], ["IMMINENT", "EARLY_RISK", "WATCHLIST"], default="STABLE")
        result["decision_reason"] = np.where(terminal & prediction, "terminal_multi_service_collapse",
                                            np.where(suppressed, "borderline_false_positive_guard",
                                                     np.where(prediction, "calibrated_model", "below_threshold")))
        return result

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "UnifiedChurnEngine":
        return joblib.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified 30-day telecom churn engine")
    parser.add_argument("mode", choices=("train", "score")); parser.add_argument("input_csv"); parser.add_argument("--artifact", default="unified_churn_engine.joblib")
    parser.add_argument("--output", default="unified_churn_scores.csv"); parser.add_argument("--precision-floor", type=float, default=.70)
    args = parser.parse_args(); logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = pd.read_csv(args.input_csv)
    if args.mode == "train":
        engine = UnifiedChurnEngine(EngineConfig(precision_floor=args.precision_floor)); summary = engine.fit(df); engine.save(args.artifact)
        Path(args.output).with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))
    else:
        engine = UnifiedChurnEngine.load(args.artifact); engine.score(df).to_csv(args.output, index=False); print(f"Wrote {args.output}")


if __name__ == "__main__": main()
