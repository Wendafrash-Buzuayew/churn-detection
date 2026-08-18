import numpy as np
import pytest

from churn_ranker.modeling import ChurnRanker, RankerConfig

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
