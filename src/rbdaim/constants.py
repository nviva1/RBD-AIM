"""Domain constants for the RBD epitope classifier.

These encode the immunology of the problem and must not drift: the 12 epitope
classes and the Chothia CDR residue ranges are part of the data contract shared
with the published checkpoints.
"""

from __future__ import annotations

# The 12 RBD epitope classes, in the canonical order used by the label encoding
# (`POS_class` integer == index into this tuple).
EPITOPE_CLASSES: tuple[str, ...] = (
    "A",
    "B",
    "C",
    "D1",
    "D2",
    "E1",
    "E21",
    "E22",
    "E3",
    "F1",
    "F2",
    "F3",
)
NUM_CLASSES: int = len(EPITOPE_CLASSES)

# Chothia-scheme CDR residue numbers (light and heavy). These select the
# residues whose embeddings are mean-pooled by the CDR-pooling model.
# Source: the original research code (`rbdaim.py:239-240`).
LIGHT_CDR_POSITIONS: frozenset[int] = frozenset(
    [*range(24, 35), *range(50, 57), *range(89, 98)]
)
HEAVY_CDR_POSITIONS: frozenset[int] = frozenset(
    [*range(31, 36), *range(50, 66), *range(95, 103)]
)

# Sentinel POS_class for antibodies with no epitope label.
UNLABELED = -1


def class_name(index: int) -> str:
    """Return the epitope class name for a label index."""
    return EPITOPE_CLASSES[index]


def class_index(name: str) -> int:
    """Return the label index for an epitope class name."""
    return EPITOPE_CLASSES.index(name)
