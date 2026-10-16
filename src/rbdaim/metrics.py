"""Classification metrics with bootstrapped confidence intervals.

Point estimates alone mislead when comparing runs on a few hundred labeled test
antibodies, so aggregate metrics come with bootstrap 95% CIs.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class ClassMetrics:
    """Aggregate classification metrics."""

    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    f1_macro: float
    mcc: float

    def to_dict(self) -> dict[str, float]:
        """Return the metrics as a plain dict."""
        return asdict(self)


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassMetrics:
    """Compute aggregate metrics for hard predictions."""
    return ClassMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_weighted=float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        recall_weighted=float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        f1_weighted=float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
    )


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str = "f1_macro",
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap a 95% CI for one metric by resampling test examples."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        m = class_metrics(y_true[idx], y_pred[idx])
        samples.append(getattr(m, metric))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def top_score_reliability_error(
    y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10
) -> float:
    """Binned gap between the top sigmoid score and observed accuracy.

    This is the Expected-Calibration-Error *computation* applied to the model's
    top per-class sigmoid score. It is reported as a diagnostic of how well that
    score tracks correctness, and deliberately not called "calibration error":
    the model was never calibrated, the scores are independent per-class
    log-odds rather than a normalized distribution, and a low value here is not
    a licence to read the score as P(correct).
    """
    confidences = scores.max(axis=1)
    predictions = scores.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lo, hi in itertools.pairwise(bins):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(error)
