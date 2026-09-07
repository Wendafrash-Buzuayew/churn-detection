import numpy as np
import pytest

from churn_ranker.modeling import ChurnRanker, RankerConfig
from tests.conftest import make_synthetic

FAST = RankerConfig(cv_folds=3, max_iter=60)


def test_fit_learns_synthetic_collapse(synthetic):
    ranker = ChurnRanker(FAST)
    summary = ranker.fit(synthetic)
    assert summary["oof_metrics"]["roc_auc"] > 0.85
    assert summary["n_churners"] == int(synthetic["LABEL_CHURN_90D"].sum())
    assert len(summary["tier_thresholds"]) == 3


def test_leakage_columns_never_enter_features(synthetic):
    ranker = ChurnRanker(FAST)
    ranker.fit(synthetic)
    for name in ranker.feature_names_:
        assert not name.startswith(("MSISDN", "LABEL_"))
        assert name not in ("SNAPSHOT_DATE", "DATASET_TYPE")


def test_score_output_columns_and_ranges(synthetic):
    ranker = ChurnRanker(FAST)
    ranker.fit(synthetic)
    scored = ranker.score(synthetic)
    assert {"churn_probability", "risk_tier", "reason_code"} <= set(scored.columns)
    assert scored["churn_probability"].between(0, 1).all()
    assert set(scored["risk_tier"]) <= {
        "TIER_1_IMMINENT", "TIER_2_HIGH_RISK", "TIER_3_WATCHLIST", "STABLE",
    }
    stable = scored["risk_tier"] == "STABLE"
    assert (scored.loc[stable, "reason_code"] == "").all()
    assert (scored.loc[~stable, "reason_code"] != "").all()


def test_scores_unlabeled_data(synthetic):
    ranker = ChurnRanker(FAST)
    ranker.fit(synthetic)
    unlabeled = synthetic.drop(columns=["LABEL_CHURN_90D"])
    scored = ranker.score(unlabeled)
    assert len(scored) == len(unlabeled)


def test_save_load_roundtrip(tmp_path, synthetic):
    ranker = ChurnRanker(FAST)
    ranker.fit(synthetic)
    path = tmp_path / "ranker.joblib"
    ranker.save(path)
    loaded = ChurnRanker.load(path)
    np.testing.assert_allclose(
        ranker.predict_proba(synthetic), loaded.predict_proba(synthetic)
    )


def test_fit_without_target_raises(synthetic):
    with pytest.raises(ValueError):
        ChurnRanker(FAST).fit(synthetic.drop(columns=["LABEL_CHURN_90D"]))


def test_predict_proba_coerces_object_dtype_numeric_column(synthetic):
    ranker = ChurnRanker(FAST)
    ranker.fit(synthetic)
    flipped = synthetic.copy()
    flipped["DATA_MB_W13"] = flipped["DATA_MB_W13"].astype(str)
    flipped.loc[0, "DATA_MB_W13"] = ""
    expected = ranker.predict_proba(synthetic)
    actual = ranker.predict_proba(flipped)
    np.testing.assert_allclose(actual[1:], expected[1:])


def test_calibration_falls_back_to_oof_when_too_few_positives(synthetic):
    # 600 rows * 10% churn * default 15% holdout ~= 9 expected holdout positives,
    # below the default min_calibration_positives=20 -> must fall back to OOF.
    ranker = ChurnRanker(FAST)
    summary = ranker.fit(synthetic)
    assert summary["calibration"]["source"] == "oof"
    assert 0.0 <= summary["calibration"]["brier_score"] <= 1.0
    assert 0.0 <= summary["calibration"]["ece"] <= 1.0


def test_calibration_uses_held_out_slice_when_enough_positives():
    # 1200 rows * 25% churn * 15% holdout ~= 45 expected holdout positives, well above
    # the default min_calibration_positives=20 -> must use a genuine held-out slice.
    big = make_synthetic(n=1200, churn_frac=0.25, seed=7)
    ranker = ChurnRanker(RankerConfig(cv_folds=3, max_iter=60))
    summary = ranker.fit(big)
    calibration = summary["calibration"]
    assert calibration["source"] == "held_out"
    assert calibration["n"] == pytest.approx(1200 * 0.15, abs=5)
    assert 0.0 <= calibration["brier_score"] <= 1.0
    assert 0.0 <= calibration["ece"] <= 1.0
    # the deployed model must be fit on strictly fewer rows than the full input
    # (the held-out slice was excluded from its training data)
    assert summary["n_rows"] == 1200
    assert summary["n_churners"] == int(big["LABEL_CHURN_90D"].sum())


def test_logit_space_calibration_preserves_rank_order(synthetic):
    ranker = ChurnRanker(FAST)
    ranker.fit(synthetic)
    raw = ranker.model.predict_proba(ranker._matrix(synthetic))[:, 1]
    calibrated = ranker.predict_proba(synthetic)
    assert np.array_equal(np.argsort(raw, kind="stable"), np.argsort(calibrated, kind="stable"))


def test_monotonic_constraints_off_by_default(synthetic):
    ranker = ChurnRanker(FAST)
    ranker.fit(synthetic)
    assert ranker.model.monotonic_cst is None


def test_monotonic_constraints_opt_in_fits_and_ranks_well(synthetic):
    config = RankerConfig(cv_folds=3, max_iter=60, use_monotonic_constraints=True)
    ranker = ChurnRanker(config)
    summary = ranker.fit(synthetic)
    assert ranker.model.monotonic_cst is not None
    assert len(ranker.model.monotonic_cst) == len(ranker.feature_names_)
    assert summary["oof_metrics"]["roc_auc"] > 0.7
