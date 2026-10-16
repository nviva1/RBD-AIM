"""Typed config composition."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from rbdaim.cli import config_dir, load_config
from rbdaim.config import ModelConfig, TrainConfig, from_dictconfig
from rbdaim.constants import NUM_CLASSES


def test_packaged_conf_dir_exists():
    """pip-installed CLI must find its own conf/, not die composing defaults."""
    packaged = config_dir()
    assert (packaged / "config.yaml").is_file()
    for group in ("data", "model", "train"):
        assert (packaged / group).is_dir()


def test_load_config_requires_an_explicit_data_path():
    """data.path is mandatory: no machine-specific default to silently inherit."""
    from omegaconf.errors import MissingMandatoryValue

    with pytest.raises(MissingMandatoryValue):
        load_config([])


def test_load_config_with_overrides():
    cfg = load_config(["data.path=/tmp/x.pkl", "train.epochs=9", "model=antiberty_cdr"])
    assert cfg.data.path == "/tmp/x.pkl"
    assert cfg.train.epochs == 9
    assert cfg.model.pooling == "cdr"


def test_unknown_key_raises_instead_of_being_ignored():
    """`train.epocs=10` must fail loudly, not run with the default epoch count."""
    cfg = OmegaConf.create({"data": {"path": "x.pkl"}, "train": {"epocs": 10}})
    with pytest.raises(ValueError, match="unknown key"):
        from_dictconfig(cfg)


def test_invalid_pooling_rejected():
    with pytest.raises(ValueError, match="pooling"):
        ModelConfig(pooling="attention")


def test_num_classes_is_pinned_to_the_label_space():
    with pytest.raises(ValueError, match="num_classes"):
        ModelConfig(num_classes=NUM_CLASSES + 1)


class TestFullModelAndEnsembleIdAreMutuallyExclusive:
    """ensemble_id has no slot in the full-model naming scheme.

    The published names reuse the fold slot for the replicate, so a non-zero
    ensemble_id had nowhere to go and was silently dropped.
    """

    def test_full_model_with_a_nonzero_ensemble_id_is_rejected(self):
        with pytest.raises(ValueError, match="no slot of its own"):
            TrainConfig(full_model=True, ensemble_id=1)

    def test_full_model_with_ensemble_id_zero_is_fine(self):
        assert TrainConfig(full_model=True, ensemble_id=0).full_model is True

    def test_cross_validation_still_accepts_every_replicate(self):
        """Only the full-model path lacks a replicate slot."""
        for eid in range(5):
            assert TrainConfig(full_model=False, ensemble_id=eid).ensemble_id == eid

    def test_the_names_that_used_to_collide_are_now_unreachable(self):
        """Two replicates previously produced one filename; proven here."""
        from rbdaim.trainer import checkpoint_name

        names = {checkpoint_name(0, "CLS", full=True, ensemble_id=e) for e in (0, 1, 2)}
        assert len(names) == 1  # the collision the config guard now prevents
        with pytest.raises(ValueError):
            TrainConfig(full_model=True, ensemble_id=2)
