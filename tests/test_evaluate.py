"""Evaluation and scoring semantics."""

from __future__ import annotations

import numpy as np
import pytest

from rbdaim import evaluate
from rbdaim.constants import NUM_CLASSES


def test_evaluate_reports_sane_metrics(logits_labels):
    logits, labels = logits_labels
    report = evaluate.evaluate(logits, labels)
    assert report.n == len(labels)
    assert report.metrics["accuracy"] > 0.8  # logits are separable by construction
    lo, hi = report.f1_macro_ci
    assert lo <= report.metrics["f1_macro"] <= hi


def test_class_scores_are_sigmoid_not_softmax():
    """Scores must be independent per-class sigmoids of BCE logits."""
    logits = np.array([[2.0, 2.0, 2.0, 2.0] + [0.0] * (NUM_CLASSES - 4)])
    scores = evaluate.class_scores(logits)
    # Sigmoid(2.0) ~ 0.88 for each of the four; a softmax would force sum == 1.
    np.testing.assert_allclose(scores[0, :4], 1 / (1 + np.exp(-2.0)), rtol=1e-6)
    assert scores.sum() > 1.0


def test_argmax_is_unchanged_by_the_scoring_function(logits_labels):
    """Switching softmax->sigmoid must not move any prediction."""
    logits, _ = logits_labels
    softmax = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(
        evaluate.class_scores(logits).argmax(axis=1), softmax.argmax(axis=1)
    )


def test_scores_are_not_a_normalized_distribution(logits_labels):
    """The distinguishing property: sigmoid rows do not sum to 1, softmax rows do.

    Without this, the argmax test above passes for softmax too — it holds for
    any monotone transform — so it cannot tell the two apart on its own.
    """
    logits, _ = logits_labels
    row_sums = evaluate.class_scores(logits).sum(axis=1)
    assert not np.allclose(row_sums, 1.0)
    # And each score is exactly the inverse logit of its own logit, nothing else.
    np.testing.assert_allclose(
        evaluate.class_scores(logits), 1.0 / (1.0 + np.exp(-logits)), rtol=1e-9
    )


def test_evaluate_rejects_unlabeled_rows(logits_labels):
    """An unlabeled sentinel must raise, not be scored as class 0."""
    logits, labels = logits_labels
    labels = labels.copy()
    labels[0] = -1
    with pytest.raises(ValueError, match="unlabeled"):
        evaluate.evaluate(logits, labels)


def test_evaluate_rejects_empty_and_mismatched(logits_labels):
    logits, labels = logits_labels
    with pytest.raises(ValueError, match="empty"):
        evaluate.evaluate(logits[:0], labels[:0])
    with pytest.raises(ValueError, match="mismatch"):
        evaluate.evaluate(logits[:10], labels[:20])


def test_report_renders_every_metric(logits_labels):
    logits, labels = logits_labels
    text = evaluate.evaluate(logits, labels).render()
    for key in ("accuracy", "f1_macro", "mcc", "95% CI", "reliability_error"):
        assert key in text


def test_report_round_trips_through_its_dict(logits_labels):
    logits, labels = logits_labels
    report = evaluate.evaluate(logits, labels)
    as_dict = report.to_dict()
    assert as_dict["n"] == report.n
    assert as_dict["metrics"] == report.metrics
    assert as_dict["f1_macro_ci95"] == list(report.f1_macro_ci)


class TestNonFiniteLogitsAreRejected:
    """A nan row does not score as missing — argmax prefers it."""

    def test_a_nan_logit_is_rejected(self, logits_labels):
        logits, labels = logits_labels
        logits = logits.copy()
        logits[3, 7] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            evaluate.evaluate(logits, labels)

    def test_the_error_counts_affected_rows_not_cells(self, logits_labels):
        logits, labels = logits_labels
        logits = logits.copy()
        logits[5, 0] = np.nan
        logits[5, 1] = np.nan  # same row, two cells
        logits[9, 2] = np.inf
        with pytest.raises(ValueError, match=r"^2 row\(s\)"):
            evaluate.evaluate(logits, labels)

    def test_finite_logits_are_unaffected(self, logits_labels):
        """The guard must not change any number for well-formed input."""
        logits, labels = logits_labels
        assert evaluate.evaluate(logits, labels).metrics["accuracy"] > 0.8
