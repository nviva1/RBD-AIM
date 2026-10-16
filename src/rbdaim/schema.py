"""Data contract for the labeled Fab dataset.

Validated on load so malformed data fails fast with a typed error instead of a
``KeyError`` deep inside the training loop. The contract was reverse-engineered
from and verified against the real ``dataset_production.pkl`` (1,810 rows).
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column

from rbdaim.constants import NUM_CLASSES


def build_schema(num_folds: int = 10) -> pa.DataFrameSchema:
    """Build the dataframe schema, including the per-fold split columns."""
    columns: dict[str, Column] = {
        "index": Column(nullable=False, unique=True, coerce=False),
        "POS_class": Column(
            int,
            Check.isin(range(-1, NUM_CLASSES)),
            nullable=False,
            coerce=True,
        ),
        # `good` is null for unlabeled production rows; don't coerce (NaN→int).
        "good": Column(checks=Check.isin([-1, 0, 1]), nullable=True),
        "heavy_anarci": Column(str, nullable=False),
        "light_anarci": Column(str, nullable=False),
        # CDR index arrays are stored as object cells (numpy arrays); we assert
        # presence and non-nullity, not element dtype.
        "heavy_ids_cdr": Column(object, nullable=False),
        "light_ids_cdr": Column(object, nullable=False),
        "our_test": Column(nullable=False),
    }
    # Fold-membership columns are null for rows outside the cross-validation set
    # (e.g. unlabeled production antibodies); the trainer treats null as "not in
    # this fold's validation split".
    for fold in range(num_folds):
        columns[f"cross_val_test_{fold}"] = Column(nullable=True)
    return pa.DataFrameSchema(columns, strict=False, coerce=False)


def validate(df: pd.DataFrame, num_folds: int = 10) -> pd.DataFrame:
    """Validate ``df`` against the contract.

    Raises:
        pandera.errors.SchemaError: if the dataframe violates the contract.

    """
    return build_schema(num_folds).validate(df, lazy=True)
