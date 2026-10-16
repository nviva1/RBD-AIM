"""Checkpoint remapping and load verification."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from rbdaim.constants import EPITOPE_CLASSES, NUM_CLASSES
from rbdaim.numbering import ChainNumberingError
from rbdaim.predictor import EpitopePredictor, _remap_legacy, load_checkpoint


def test_legacy_prefixes_are_remapped():
    legacy = {
        "model_l.bert.weight": torch.zeros(1),
        "model_h.bert.weight": torch.zeros(1),
        "epitopes.weight": torch.zeros(1),
        "epitopes.bias": torch.zeros(1),
    }
    out = _remap_legacy(legacy)
    assert set(out) == {
        "encoder_light.bert.weight",
        "encoder_heavy.bert.weight",
        "head.weight",
        "head.bias",
    }


def test_modern_state_dict_is_passed_through_untouched():
    modern = {"encoder_light.bert.weight": torch.zeros(1), "head.bias": torch.zeros(1)}
    assert _remap_legacy(modern) is modern


def test_load_rejects_a_checkpoint_that_matches_nothing(tmp_path):
    """A silent strict=False load is how an untrained net reports good numbers."""
    bad = tmp_path / "NN_cv_0_CLS_full.pth"
    torch.save({"model_state": {"not_a_real_layer.weight": torch.zeros(3)}}, bad)
    with pytest.raises(RuntimeError, match="incompatible checkpoint"):
        load_checkpoint(bad, pooling="cls")


def test_load_rejects_an_unwrapped_rich_dict(tmp_path):
    """A rich dict left wrapped would load zero weights under strict=False."""
    bad = tmp_path / "NN_cv_0_CLS_full.pth"
    torch.save({"model_state": {"model_state": {"head.bias": torch.zeros(12)}}}, bad)
    with pytest.raises(RuntimeError, match="incompatible checkpoint"):
        load_checkpoint(bad, pooling="cls")


class _StubModel:
    """Returns fixed logits, so `_run`'s own logic is what is under test."""

    def __init__(self, logits):
        self._logits = torch.as_tensor(logits, dtype=torch.float32).reshape(1, -1)

    def __call__(self, **kwargs):
        return {"logits": self._logits, "embeddings": torch.zeros(1, 4)}


def _predictor(logits):
    # A stub standing in for the real model: `_run` is what is under test.
    return EpitopePredictor([_StubModel(logits)], torch.device("cpu"))  # type: ignore[list-item]


class TestNonFiniteLogitsAreRefused:
    """argmax prefers nan, so a nan row looks like a confident prediction."""

    def test_all_nan_logits_raise_instead_of_predicting_the_first_class(self):
        pred = _predictor([float("nan")] * NUM_CLASSES)
        with pytest.raises(ValueError, match="non-finite"):
            pred._run({})

    def test_a_single_nan_does_not_hijack_a_clear_winner(self):
        """Without the guard this returns the nan's class, not the 99.0 one."""
        logits = [0.0] * NUM_CLASSES
        logits[3] = 99.0
        logits[7] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            _predictor(logits)._run({})

    def test_infinite_logits_are_refused_too(self):
        logits = [0.0] * NUM_CLASSES
        logits[2] = float("inf")
        with pytest.raises(ValueError, match="non-finite"):
            _predictor(logits)._run({})

    def test_finite_logits_still_predict_the_argmax(self):
        logits = [0.0] * NUM_CLASSES
        logits[5] = 9.0
        assert _predictor(logits)._run({}).predicted_class == EPITOPE_CLASSES[5]


class TestPredictNumberedValidatesItsCdrIndices:
    """`predict_numbered` skips `number_chain`, so it must guard the row itself."""

    def _row(self, light_cdr, heavy_cdr):
        return pd.Series(
            {
                "light_anarci": "D" * 20,
                "heavy_anarci": "E" * 20,
                "light_ids_cdr": light_cdr,
                "heavy_ids_cdr": heavy_cdr,
            }
        )

    def test_an_empty_light_cdr_is_rejected(self):
        pred = _predictor([0.0] * NUM_CLASSES)
        with pytest.raises(ChainNumberingError, match="no CDR positions"):
            pred.predict_numbered(self._row([], [1, 2, 3]))

    def test_an_empty_heavy_cdr_is_rejected(self):
        pred = _predictor([0.0] * NUM_CLASSES)
        with pytest.raises(ChainNumberingError, match="no CDR positions"):
            pred.predict_numbered(self._row([1, 2, 3], []))

    def test_an_out_of_range_cdr_index_is_rejected(self):
        """Index 21 into a 20-residue chain gathers padding, not a residue."""
        pred = _predictor([0.0] * NUM_CLASSES)
        with pytest.raises(ChainNumberingError, match="outside"):
            pred.predict_numbered(self._row([1, 2], [1, 21]))
