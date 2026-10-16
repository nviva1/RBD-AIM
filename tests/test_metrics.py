"""Metric computation and bootstrap CIs."""

from __future__ import annotations

import numpy as np

from rbdaim.metrics import bootstrap_ci, class_metrics, top_score_reliability_error


def test_perfect_predictions_score_one():
    y = np.array([0, 1, 2, 3, 0, 1])
    m = class_metrics(y, y)
    assert m.accuracy == 1.0
    assert m.f1_macro == 1.0
    assert m.mcc == 1.0


def test_metrics_degrade_with_errors():
    y_true = np.array([0, 1, 2, 3, 0, 1])
    y_pred = np.array([0, 1, 2, 3, 1, 0])
    m = class_metrics(y_true, y_pred)
    assert 0.5 < m.accuracy < 1.0
    assert m.mcc < 1.0


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, size=200)
    y_pred = y_true.copy()
    y_pred[:40] = (y_pred[:40] + 1) % 4
    point = class_metrics(y_true, y_pred).f1_macro
    lo, hi = bootstrap_ci(y_true, y_pred, metric="f1_macro", seed=0)
    assert lo <= point <= hi
    assert hi - lo < 0.3


def test_bootstrap_ci_is_deterministic_for_a_seed():
    y = np.array([0, 1, 2, 3] * 20)
    assert bootstrap_ci(y, y, seed=3) == bootstrap_ci(y, y, seed=3)


def test_reliability_error_is_small_when_scores_track_accuracy():
    n = 400
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 4, size=n)
    scores = np.full((n, 4), 0.02)
    scores[np.arange(n), y_true] = 0.95  # confident and right
    assert top_score_reliability_error(y_true, scores) < 0.1


def test_reliability_error_is_large_when_confident_and_wrong():
    n = 200
    y_true = np.zeros(n, dtype=int)
    scores = np.full((n, 4), 0.01)
    scores[:, 1] = 0.99  # confident on the wrong class every time
    assert top_score_reliability_error(y_true, scores) > 0.8
