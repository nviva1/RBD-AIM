"""Data contract validation."""

from __future__ import annotations

import pandera.errors
import pytest

from rbdaim import schema

from .conftest import make_frame


def test_valid_frame_passes():
    frame = make_frame()
    assert len(schema.validate(frame, num_folds=2)) == len(frame)


def test_missing_required_column_fails():
    frame = make_frame().drop(columns=["heavy_anarci"])
    with pytest.raises(pandera.errors.SchemaErrors):
        schema.validate(frame, num_folds=2)


def test_out_of_range_class_fails():
    frame = make_frame()
    frame.loc[0, "POS_class"] = 99
    with pytest.raises(pandera.errors.SchemaErrors):
        schema.validate(frame, num_folds=2)


def test_unlabeled_sentinel_is_allowed():
    frame = make_frame(unlabeled=5)
    schema.validate(frame, num_folds=2)


def test_duplicate_index_fails():
    frame = make_frame()
    frame.loc[1, "index"] = frame.loc[0, "index"]
    with pytest.raises(pandera.errors.SchemaErrors):
        schema.validate(frame, num_folds=2)
