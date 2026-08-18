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
