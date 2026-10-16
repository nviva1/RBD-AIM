"""Turn predicted logits into metrics.

Evaluation never trains. It takes logits that already exist — produced by
running checkpoints over held-out rows — and scores them, with bootstrap
confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rbdaim.metrics import bootstrap_ci, class_metrics, top_score_reliability_error


@dataclass(frozen=True)
class EvalReport:
    """Full evaluation report for a run."""

    metrics: dict[str, float]
    f1_macro_ci: tuple[float, float]
    reliability_error: float
    n: int

    def to_dict(self) -> dict[str, object]:
        """Return the report as a JSON-serializable dict."""
        return {
            "metrics": self.metrics,
            "f1_macro_ci95": list(self.f1_macro_ci),
            "reliability_error": self.reliability_error,
            "n": self.n,
        }

    def render(self) -> str:
        """Return a human-readable summary block."""
        lines = [f"n = {self.n}"]
        lines += [f"{k:>20}: {v:.4f}" for k, v in self.metrics.items()]
        lo, hi = self.f1_macro_ci
        lines.append(f"{'f1_macro 95% CI':>20}: [{lo:.4f}, {hi:.4f}]")
        lines.append(f"{'reliability_error':>20}: {self.reliability_error:.4f}")
        return "\n".join(lines)


def class_scores(logits: np.ndarray) -> np.ndarray:
    """Per-class sigmoid scores for BCE-trained logits.

    The model is trained with per-class ``binary_cross_entropy_with_logits``, so
    each logit is an independent log-odds and sigmoid is its inverse. Applying a
    row-wise softmax instead would produce numbers that look like a probability
    distribution but are not the model's output; argmax is unchanged either way,
    the reported scores are not.
    """
    result: np.ndarray = 1.0 / (1.0 + np.exp(-logits))
    return result


def evaluate(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 0,
) -> EvalReport:
    """Build an :class:`EvalReport` from predicted logits and integer labels.

    Args:
        logits: ``(n, num_classes)`` predicted logits.
        labels: ``(n,)`` integer ground-truth classes. Must all be >= 0;
            unlabeled rows are filtered upstream by ``data.labeled_mask``.
        seed: Bootstrap seed.

    Raises:
        ValueError: if ``labels`` contains unlabeled sentinels, if shapes
            disagree, or if ``logits`` holds a non-finite value.

    """
    if len(logits) != len(labels):
        raise ValueError(f"shape mismatch: {len(logits)} logits, {len(labels)} labels")
    if len(labels) == 0:
        raise ValueError("nothing to evaluate: empty prediction set")
    if not np.isfinite(logits).all():
        n_rows = int((~np.isfinite(logits)).any(axis=1).sum())
        raise ValueError(
            f"{n_rows} row(s) hold a non-finite logit. These do not score as "
            "missing: argmax prefers nan, so they would silently contribute "
            "wrong predictions to every metric below, and the binned "
            "reliability error would drop them while still dividing by the "
            "full row count. Usual cause: an empty or out-of-range CDR index."
        )
    if (labels < 0).any():
        n_bad = int((labels < 0).sum())
        raise ValueError(
            f"{n_bad} unlabeled row(s) reached evaluate(). Filter them with "
            "rbdaim.data.labeled_mask before scoring — an unlabeled row has no "
            "ground truth to be right or wrong about."
        )
    scores = class_scores(logits)
    preds = logits.argmax(axis=1)
    m = class_metrics(labels, preds)
    lo, hi = bootstrap_ci(labels, preds, metric="f1_macro", seed=seed)
    rel = top_score_reliability_error(labels, scores)
    return EvalReport(m.to_dict(), (lo, hi), rel, len(labels))
