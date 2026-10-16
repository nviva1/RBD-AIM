"""RBD-AIM classifier: SARS-CoV-2 RBD antibody epitope prediction.

A refactor of the AntiBERTy-based paired-chain epitope classifier from
Terekhov SS, Ivanisenko NV et al., *Mining antibody functionality via AI-guided
structural landscape profiling*, Nature Communications 2026.
https://www.nature.com/articles/s41467-026-70553-6

The public API is intentionally small; see :mod:`rbdaim.cli` for the
command-line entrypoints.
"""

from rbdaim.constants import EPITOPE_CLASSES, NUM_CLASSES

__version__ = "1.0.0"

__all__ = ["EPITOPE_CLASSES", "NUM_CLASSES", "__version__"]
