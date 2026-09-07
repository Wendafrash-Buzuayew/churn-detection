import numpy as np
import pytest

from churn_ranker import evaluation


def test_lift_table_hand_computed():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    table = evaluation.lift_table(y, scores, top_fractions=(0.1, 0.3))
    top10 = table.iloc[0]
    assert top10["contacted"] == 1
    assert top10["churners_caught"] == 1
    assert top10["precision"] == pytest.approx(1.0)
    assert top10["recall"] == pytest.approx(0.5)
    assert top10["lift"] == pytest.approx(5.0)  # precision 1.0 / base rate 0.2
    top30 = table.iloc[1]
    assert top30["contacted"] == 3
    assert top30["churners_caught"] == 2
    assert top30["recall"] == pytest.approx(1.0)


def test_ranking_metrics_keys_and_base_rate():
    y = np.array([1, 0, 0, 0])
    scores = np.array([0.9, 0.2, 0.1, 0.4])
    metrics = evaluation.ranking_metrics(y, scores)
    assert metrics["n"] == 4
    assert metrics["positives"] == 1
    assert metrics["base_rate"] == pytest.approx(0.25)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert 0.0 < metrics["pr_auc"] <= 1.0


def test_single_class_returns_none_aucs():
    metrics = evaluation.ranking_metrics(np.zeros(5, dtype=int), np.linspace(0, 1, 5))
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None


def test_lift_table_empty_input_returns_empty_frame():
    table = evaluation.lift_table(np.array([]), np.array([]))
    assert len(table) == 0
    assert list(table.columns) == ["top_fraction", "contacted", "churners_caught", "precision", "recall", "lift"]


def test_brier_score_hand_computed():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    # (0.1-0)^2 + (0.4-0)^2 + (0.6-1)^2 + (0.9-1)^2 = 0.01+0.16+0.16+0.01 = 0.34 / 4
    assert evaluation.brier_score(y, scores) == pytest.approx(0.085)


def test_brier_score_perfect_predictions_is_zero():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.0, 1.0, 0.0, 1.0])
    assert evaluation.brier_score(y, scores) == pytest.approx(0.0)


def test_brier_score_empty_input_returns_zero():
    assert evaluation.brier_score(np.array([]), np.array([])) == 0.0


def test_expected_calibration_error_hand_computed():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    # bin [0, 0.5]: scores 0.1, 0.4 -> mean conf 0.25, mean actual 0.0 -> |diff| 0.25, weight 0.5
    # bin (0.5, 1]: scores 0.6, 0.9 -> mean conf 0.75, mean actual 1.0 -> |diff| 0.25, weight 0.5
    # ECE = 0.5*0.25 + 0.5*0.25 = 0.25
    assert evaluation.expected_calibration_error(y, scores, n_bins=2) == pytest.approx(0.25)


def test_expected_calibration_error_perfect_calibration_is_zero():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < scores).astype(int)
    assert evaluation.expected_calibration_error(y, scores, n_bins=10) < 0.05


def test_expected_calibration_error_empty_input_returns_zero():
    assert evaluation.expected_calibration_error(np.array([]), np.array([])) == 0.0
