"""Seeding and device resolution."""

from __future__ import annotations

import numpy as np
import torch

from rbdaim.torch_utils import resolve_device, seed_everything


def test_seeding_makes_draws_reproducible():
    seed_everything(123)
    a = (torch.randn(5), np.random.rand(5))
    seed_everything(123)
    b = (torch.randn(5), np.random.rand(5))
    assert torch.equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_different_seeds_diverge():
    seed_everything(1)
    a = torch.randn(5)
    seed_everything(2)
    assert not torch.equal(a, torch.randn(5))


def test_explicit_device_is_honoured():
    assert resolve_device("cpu").type == "cpu"


def test_auto_resolves_to_an_available_device():
    assert resolve_device("auto").type in {"cuda", "mps", "cpu"}
