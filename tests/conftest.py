"""Shared fixtures.

Everything here is synthetic. The suite must run with no AntiBERTy weights, no
dataset and no network, because a test that only passes on the author's laptop
is not a test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rbdaim.constants import NUM_CLASSES


def make_frame(n: int = 24, n_folds: int = 2, unlabeled: int = 3) -> pd.DataFrame:
    """Build a synthetic frame matching the production data contract.

    Args:
        n: Total rows.
        n_folds: Number of cross-validation split columns.
        unlabeled: How many rows carry ``POS_class == -1``.

    """
    rng = np.random.default_rng(0)
    classes = [int(i % NUM_CLASSES) for i in range(n)]
    for i in range(unlabeled):
        classes[-(i + 1)] = -1

    frame = pd.DataFrame(
        {
            "index": list(range(n)),
            "POS_class": classes,
            "good": [
                1 if c >= 0 and i % 3 == 0 else (0 if c >= 0 else -1)
                for i, c in enumerate(classes)
            ],
            "heavy_anarci": [
                "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 30)) for _ in range(n)
            ],
            "light_anarci": [
                "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 28)) for _ in range(n)
            ],
            "heavy_ids_cdr": [np.array([3, 4, 5], dtype=np.int64) for _ in range(n)],
            "light_ids_cdr": [np.array([2, 3, 4], dtype=np.int64) for _ in range(n)],
            "our_test": [i % 7 == 0 for i in range(n)],
        }
    )
    for fold in range(n_folds):
        frame[f"cross_val_test_{fold}"] = [i % n_folds == fold for i in range(n)]
    assert isinstance(frame, pd.DataFrame)
    return frame


@pytest.fixture
def frame() -> pd.DataFrame:
    """Return a synthetic production-contract dataframe."""
    return make_frame()


@pytest.fixture
def logits_labels() -> tuple[np.ndarray, np.ndarray]:
    """Separable synthetic logits and their integer labels."""
    rng = np.random.default_rng(1)
    labels = rng.integers(0, NUM_CLASSES, size=200)
    logits = rng.normal(0, 0.5, size=(200, NUM_CLASSES))
    logits[np.arange(200), labels] += 3.0  # mostly-correct predictions
    return logits, labels
