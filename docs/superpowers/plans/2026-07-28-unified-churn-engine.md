# Unified Churn Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a runnable, calibrated hybrid telecom churn pipeline targeting `LABEL_CHURN_30D`.

**Architecture:** A single modular Python module resolves safe training/scoring modes, produces vectorized domain/rule features, fits a gradient-booster and calibration layer, applies precision-first thresholding plus borderline false-positive guards, and persists artifacts/reports. Pytest uses synthetic data to verify leakage prevention, OOT-only scoring, and guard behavior.

**Tech Stack:** Python 3, pandas, NumPy, scikit-learn, joblib, pytest; optional LightGBM/XGBoost/CatBoost.

## Global Constraints

- `LABEL_CHURN_30D` is the only supervised target.
- Exclude every `LABEL_CHURN_*`, IDs, `DATASET_TYPE`, and post-outcome fields from features.
- Fit preprocessing, clipping, calibration, and threshold selection only on training data.
- OOT-only/unlabeled files must score only; never train or stratify them.
- Avoid row-wise Pandas apply and NumPy `apply_along_axis` in feature engineering.
- Return Precision, Recall, F1, ROC-AUC, PR-AUC, and false-positive guard impact for labeled evaluation data.

---

### Task 1: Data contract and vectorized hybrid features

**Files:**
- Create: `unified_churn_engine.py`
- Create: `tests/test_unified_churn_engine.py`

**Interfaces:**
- Produces: `EngineConfig`, `HybridFeatureBuilder`, `resolve_target`, `resolve_split`.
- `HybridFeatureBuilder.transform(df: pd.DataFrame) -> pd.DataFrame` returns numeric model features plus `RULE_*` and `GUARD_*` columns.

- [ ] **Step 1: Write failing schema and feature tests**

```python
def test_feature_builder_creates_terminal_collapse_and_guard():
    df = pd.DataFrame({"AON": [500], "DATA_MB_W10": [100], "DATA_MB_W11": [100],
                       "DATA_MB_W12": [100], "DATA_MB_W13": [0],
                       "OG_VOICE_MIN_W10": [10], "OG_VOICE_MIN_W11": [10],
                       "OG_VOICE_MIN_W12": [10], "OG_VOICE_MIN_W13": [0],
                       "BUNDLE_CNT_W10": [1], "BUNDLE_CNT_W11": [1],
                       "BUNDLE_CNT_W12": [1], "BUNDLE_CNT_W13": [0]})
    result = HybridFeatureBuilder().transform(df)
    assert result.loc[0, "RULE_TERMINAL_MULTI_SERVICE"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_unified_churn_engine.py::test_feature_builder_creates_terminal_collapse_and_guard -v`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement schema normalization and vectorized features**

```python
def terminal_zero_run(zero_mask: np.ndarray) -> np.ndarray:
    run = np.zeros(zero_mask.shape[0], dtype=np.int8)
    for week in range(zero_mask.shape[1]):
        run = np.where(zero_mask[:, week], run + 1, 0)
    return run
```

Implement W13-baseline ratios, weighted W10-W13 decay, terminal zero runs, service breadth/divergence, recovery, recharge cliffs, and `RULE_*`/`GUARD_*` signals with NumPy matrices.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_unified_churn_engine.py -v`
Expected: PASS for feature-contract tests.

- [ ] **Step 5: Commit**

```bash
git add unified_churn_engine.py tests/test_unified_churn_engine.py
git commit -m "feat: add vectorized hybrid churn features"
```

### Task 2: Leakage-safe training, calibration, and threshold selection

**Files:**
- Modify: `unified_churn_engine.py`
- Modify: `tests/test_unified_churn_engine.py`

**Interfaces:**
- Consumes: `HybridFeatureBuilder.transform`.
- Produces: `UnifiedChurnEngine.fit(df) -> EvaluationResult`, `UnifiedChurnEngine.predict_proba(df) -> np.ndarray`, `select_precision_threshold(y, p, floor) -> float`.

- [ ] **Step 1: Write failing split and leakage tests**

```python
def test_labels_and_split_tag_never_enter_model_features(labeled_df):
    engine = UnifiedChurnEngine(EngineConfig(cv_folds=2))
    engine.fit(labeled_df)
    assert all(not c.startswith("LABEL_CHURN_") for c in engine.feature_columns_)
    assert "DATASET_TYPE" not in engine.feature_columns_
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_unified_churn_engine.py::test_labels_and_split_tag_never_enter_model_features -v`
Expected: FAIL because fitting is not implemented.

- [ ] **Step 3: Implement split handling, OOF calibration, and thresholding**

```python
def select_precision_threshold(y: np.ndarray, p: np.ndarray, floor: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y, p)
    viable = np.flatnonzero(precision[:-1] >= floor)
    index = viable[np.argmax(recall[:-1][viable])] if viable.size else int(np.argmax(precision[:-1]))
    return float(thresholds[index])
```

Use TRAIN/TEST, then TRAIN/OOT, then stratified fallback only with two classes; fit clipping/imputation on train folds; derive calibration predictions via `StratifiedKFold`; use logistic sigmoid calibration; refit the final model on all training rows.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_unified_churn_engine.py -v`
Expected: PASS including split and leakage tests.

- [ ] **Step 5: Commit**

```bash
git add unified_churn_engine.py tests/test_unified_churn_engine.py
git commit -m "feat: add calibrated precision-first churn training"
```

### Task 3: Decision guards, reporting, persistence, and CLI

**Files:**
- Modify: `unified_churn_engine.py`
- Modify: `tests/test_unified_churn_engine.py`

**Interfaces:**
- Produces: `UnifiedChurnEngine.score(df) -> pd.DataFrame`, `save(path)`, `load(path)`, and CLI subcommands `train` and `score`.

- [ ] **Step 1: Write failing guard and OOT-score tests**

```python
def test_terminal_collapse_is_not_suppressed(engine, terminal_df):
    scored = engine.score(terminal_df)
    assert not scored.loc[0, "false_positive_suppressed"]

def test_unlabeled_oot_scores_without_split(engine, oot_df):
    scored = engine.score(oot_df.drop(columns=["LABEL_CHURN_30D"]))
    assert "churn_probability" in scored
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_unified_churn_engine.py::test_terminal_collapse_is_not_suppressed tests/test_unified_churn_engine.py::test_unlabeled_oot_scores_without_split -v`
Expected: FAIL because scoring policy is incomplete.

- [ ] **Step 3: Implement decision policy and artifact I/O**

```python
suppressed = candidate & borderline & guard & ~terminal_override
prediction = candidate & ~suppressed
```

Emit calibrated probability, rule score, guard/reason columns, risk tier, and prediction. Save a joblib dictionary containing configuration, fitted builder state, feature list, model, calibrator, and threshold. Write JSON metrics and CSV outputs with confusion matrix and guard impact.

- [ ] **Step 4: Run complete verification**

Run: `python -m py_compile unified_churn_engine.py; pytest tests/test_unified_churn_engine.py -v`
Expected: compile succeeds and all tests pass.

- [ ] **Step 5: Commit**

```bash
git add unified_churn_engine.py tests/test_unified_churn_engine.py
git commit -m "feat: add production unified churn scoring engine"
```
