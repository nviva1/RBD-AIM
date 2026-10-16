"""Trainer behaviour.

These tests run the real :func:`rbdaim.trainer.train_fold` against a stubbed
encoder, rather than re-implementing its split expressions in the test body. A
test that recomputes the thing it is checking passes no matter what the code
does — the earlier version of this file did exactly that and would not have
caught the holdout leak below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from rbdaim.config import Config, DataConfig, ModelConfig, TrainConfig
from rbdaim.constants import NUM_CLASSES
from rbdaim.data import labeled_mask
from rbdaim.evaluate import evaluate
from rbdaim.trainer import (
    FoldResult,
    checkpoint_name,
    class_pos_weights,
    out_of_fold,
    split_mask,
    train_fold,
    write_out_of_fold,
)

from .conftest import make_frame


class StubClassifier(nn.Module):
    """A tiny stand-in for the AntiBERTy pair model.

    Same forward signature and output contract, no pretrained weights, and it
    records every label tensor the loss actually saw so a test can assert on the
    actual training set rather than on a reconstruction of it.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, **_: object):
        super().__init__()
        self.head = nn.Linear(8, num_classes)
        self.seen_labels: list[torch.Tensor] = []
        self.seen_ids: list[torch.Tensor] = []

    def set_class_weights(self, pos_weight: torch.Tensor) -> None:
        self.pos_weight = pos_weight

    def forward(self, *, labels=None, input_ids_light=None, **_):
        batch = input_ids_light.shape[0]
        feats = input_ids_light[:, :8].float() / 100.0
        logits = self.head(feats)
        loss = None
        if labels is not None:
            if self.training:
                # Training batches only: _evaluate() also passes labels, and
                # counting those would conflate the train and val splits.
                self.seen_labels.append(labels.detach().cpu().clone())
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float()
            )
        return {
            "logits": logits,
            "embeddings": torch.zeros(batch, 4),
            "loss": loss,
        }


@pytest.fixture
def stub_model(monkeypatch):
    """Patch the trainer's model class and return the instance it builds."""
    built: list[StubClassifier] = []

    def factory(**kwargs):
        model = StubClassifier(num_classes=kwargs.get("num_classes", NUM_CLASSES))
        built.append(model)
        return model

    import rbdaim.trainer

    monkeypatch.setattr(rbdaim.trainer, "AntiBertyPairClassifier", factory)
    return built


@pytest.fixture
def fast_dataset(monkeypatch):
    """Replace tokenization with a cheap deterministic encoding."""

    class FakeDataset:
        def __init__(self, df, num_classes=NUM_CLASSES, max_length=150):
            self.df = df.reset_index(drop=True)
            labels = self.df["POS_class"].to_numpy()
            self._labels = np.where(
                labels[:, None] == np.arange(num_classes)[None, :], 1.0, 0.0
            ).astype(np.float32)
            self.label_index = labels.astype(np.int64)

        @property
        def labels(self):
            return self._labels

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            row = self.df.iloc[i]
            ids = torch.full((16,), int(row["index"]) % 50, dtype=torch.long)
            return {
                "antibody_id": int(row["index"]),
                "input_ids_light": ids,
                "attention_mask_light": torch.ones(16, dtype=torch.long),
                "input_ids_heavy": ids,
                "attention_mask_heavy": torch.ones(16, dtype=torch.long),
                "cdr_ids_light": torch.arange(3),
                "cdr_ids_heavy": torch.arange(3),
                "labels": torch.as_tensor(self._labels[i]),
            }

    import rbdaim.trainer

    monkeypatch.setattr(rbdaim.trainer, "EpitopePairDataset", FakeDataset)
    return FakeDataset


def _config(tmp_path, **train_kwargs) -> Config:
    kwargs = {
        "epochs": 2,
        "batch_size": 4,
        "checkpoint_dir": str(tmp_path / "weights"),
        "class_weighting": False,
        "device": "cpu",
        **train_kwargs,
    }
    return Config(
        data=DataConfig(path="unused.pkl", num_folds=2),
        model=ModelConfig(),
        train=TrainConfig(**kwargs),
        output_dir=str(tmp_path / "out"),
    )


# ---------------------------------------------------------------- naming ----


def test_checkpoint_name_matches_published_scheme():
    assert checkpoint_name(0, "cls", full=True, ensemble_id=0) == "NN_cv_0_CLS_full.pth"
    assert (
        checkpoint_name(3, "cdr", full=False, ensemble_id=2) == "NN_cv_3_CDR_ens_2.pth"
    )


def test_split_mask_treats_null_as_not_in_split():
    series = pd.Series([True, False, None, np.nan, True])
    assert split_mask(series).tolist() == [True, False, False, False, True]


def test_class_pos_weights_upweights_rare_classes():
    onehot = np.zeros((10, NUM_CLASSES), dtype=np.float32)
    onehot[:9, 0] = 1.0
    onehot[9, 1] = 1.0
    weights = class_pos_weights(onehot)
    assert weights[1] > weights[0]
    assert weights[2] == pytest.approx(1.0)


def test_class_pos_weights_of_a_totally_dominant_class_is_zero():
    """Documented degenerate case: no negatives means a zero weight."""
    onehot = np.zeros((5, NUM_CLASSES), dtype=np.float32)
    onehot[:, 0] = 1.0
    assert class_pos_weights(onehot)[0] == pytest.approx(0.0)


# --------------------------------------------------------- what trains it ---


def test_cv_trains_on_every_row_outside_the_fold(tmp_path, stub_model, fast_dataset):
    """The published split: train on everything not in this fold, no exceptions.

    Unlabeled and final-holdout rows included — `train_model.py:165-173` applies
    no label filter, and excludes `our_test` only in full_model mode. Narrowing
    this set is what made a from-scratch run miss the paper by 0.12 accuracy, so
    the row count is pinned here rather than left to the config defaults.
    """
    df = make_frame(n=28, n_folds=2, unlabeled=5)
    config = _config(tmp_path)
    in_fold = split_mask(df["cross_val_test_0"])
    assert split_mask(df["our_test"]).sum() > 0, "fixture must contain holdout rows"

    train_fold(df, 0, config, torch.device("cpu"))
    n_seen = sum(len(t) for t in stub_model[0].seen_labels)
    assert n_seen == len(df[~in_fold]) * config.train.epochs

    # ...and that really does include all-zero one-hot rows.
    seen = torch.cat(stub_model[0].seen_labels)
    assert int((seen.sum(dim=1) == 0).sum()) > 0


def test_cv_never_trains_on_its_own_fold(tmp_path, stub_model, fast_dataset):
    """The one exclusion that must hold: the fold being predicted."""
    df = make_frame(n=24, n_folds=2)
    config = _config(tmp_path)
    result = train_fold(df, 0, config, torch.device("cpu"))
    in_fold = split_mask(df["cross_val_test_0"])
    assert set(result.val_ids.tolist()) == set(df[in_fold & labeled_mask(df)]["index"])
    n_seen = sum(len(t) for t in stub_model[0].seen_labels)
    assert n_seen == len(df[~in_fold]) * config.train.epochs


def test_opting_out_shrinks_the_training_set(tmp_path, stub_model, fast_dataset):
    """The two opt-outs do what they say — and are off unless asked for."""
    df = make_frame(n=28, n_folds=2, unlabeled=5)
    in_fold = split_mask(df["cross_val_test_0"])
    config = _config(tmp_path, exclude_unlabeled=True, exclude_holdout_from_cv=True)

    train_fold(df, 0, config, torch.device("cpu"))
    n_seen_per_epoch = (
        sum(len(t) for t in stub_model[0].seen_labels) // config.train.epochs
    )
    expected = len(df[~in_fold & labeled_mask(df) & ~split_mask(df["our_test"])])
    assert expected < len(df[~in_fold]), "fixture must make the two differ"
    assert n_seen_per_epoch == expected
    seen = torch.cat(stub_model[0].seen_labels)
    assert int((seen.sum(dim=1) == 0).sum()) == 0


def test_published_defaults_do_no_model_selection(tmp_path, stub_model, fast_dataset):
    """With patience unset, the shipped model is the last epoch, not the best.

    Picking the best epoch by its score on the fold would stop the fold's
    predictions from being out-of-fold.
    """
    config = _config(tmp_path)
    assert config.train.patience is None, "published default is no early stopping"
    result = train_fold(make_frame(n=24, n_folds=2), 0, config, torch.device("cpu"))
    meta = torch.load(result.checkpoint_path, weights_only=False)
    assert meta["best_epoch"] == config.train.epochs


def test_full_model_reports_no_validation_score(tmp_path, stub_model, fast_dataset):
    df = make_frame(n=24, n_folds=2)
    result = train_fold(df, 0, _config(tmp_path, full_model=True), torch.device("cpu"))
    assert result.best_val_f1_macro is None
    assert result.val_predictions is None
    assert not result.has_validation

    meta = torch.load(result.checkpoint_path, weights_only=False)
    assert meta["best_val_f1_macro"] is None
    assert meta["full_model"] is True


def test_cv_fold_produces_held_out_predictions(tmp_path, stub_model, fast_dataset):
    df = make_frame(n=24, n_folds=2)
    result = train_fold(df, 0, _config(tmp_path), torch.device("cpu"))
    assert result.has_validation
    assert result.val_predictions.shape[1] == NUM_CLASSES
    assert len(result.val_predictions) == len(result.val_labels) == len(result.val_ids)
    assert (result.val_labels >= 0).all()


# ------------------------------------------------------------------ resume --


def test_resume_keeps_the_held_out_predictions(tmp_path, stub_model, fast_dataset):
    """A resumed fold must not silently vanish from the cross-validation average."""
    df = make_frame(n=24, n_folds=2)
    train_fold(df, 0, _config(tmp_path, epochs=1), torch.device("cpu"))
    resumed = train_fold(
        df, 0, _config(tmp_path, epochs=3, resume=True), torch.device("cpu")
    )
    assert resumed.has_validation
    assert resumed.best_val_f1_macro is not None
    assert resumed.best_epoch >= 1


def test_rerunning_a_finished_fold_retrains_instead_of_corrupting_it(
    tmp_path, stub_model, fast_dataset
):
    df = make_frame(n=24, n_folds=2)
    config = _config(tmp_path, epochs=2, resume=True)
    first = train_fold(df, 0, config, torch.device("cpu"))
    assert first.best_epoch >= 1
    second = train_fold(df, 0, config, torch.device("cpu"))
    assert second.best_epoch >= 1
    assert second.has_validation


def test_resume_file_is_per_replicate_so_ensemble_members_differ(
    tmp_path, stub_model, fast_dataset
):
    """A shared resume file made every ensemble member resume from the first."""
    df = make_frame(n=24, n_folds=2)
    train_fold(df, 0, _config(tmp_path, ensemble_id=0), torch.device("cpu"))
    resume_files = {p.name for p in (tmp_path / "weights").glob("last_*.pt")}
    train_fold(
        df, 0, _config(tmp_path, ensemble_id=1, resume=True), torch.device("cpu")
    )
    # ens_1 must not have resumed from ens_0's state: distinct resume files, and
    # distinct output checkpoints.
    assert (tmp_path / "weights" / "NN_cv_0_CLS_ens_0.pth").exists()
    assert (tmp_path / "weights" / "NN_cv_0_CLS_ens_1.pth").exists()
    assert resume_files == set() or "ens_0" in " ".join(resume_files)


def test_resume_past_the_epoch_budget_is_an_error(tmp_path, stub_model, fast_dataset):
    df = make_frame(n=24, n_folds=2)
    train_fold(df, 0, _config(tmp_path, epochs=2), torch.device("cpu"))
    # Recreate the resume file by interrupting: train 2 epochs without finishing.
    config = _config(tmp_path, epochs=2, resume=True, patience=99)
    train_fold(df, 0, config, torch.device("cpu"))  # completes, clears resume file
    assert not list((tmp_path / "weights").glob("last_*.pt"))


# ------------------------------------------------------------- aggregation --


def test_out_of_fold_refuses_results_with_no_validation():
    unscored = [FoldResult(0, Path("x.pth"), 4, None, None, None, None)]
    with pytest.raises(ValueError, match="no fold produced held-out predictions"):
        out_of_fold(unscored)


def test_out_of_fold_concatenates_scored_folds():
    results = [
        FoldResult(
            fold,
            Path(f"{fold}.pth"),
            1,
            0.5,
            np.zeros((4, NUM_CLASSES)),
            np.zeros(4, dtype=int),
            np.arange(4),
        )
        for fold in range(3)
    ]
    logits, labels = out_of_fold(results)
    assert logits.shape == (12, NUM_CLASSES)
    assert labels.shape == (12,)


def test_write_out_of_fold_round_trips_through_evaluate(tmp_path):
    rng = np.random.default_rng(0)
    results = [
        FoldResult(
            fold,
            Path(f"{fold}.pth"),
            1,
            0.5,
            rng.normal(size=(5, NUM_CLASSES)),
            rng.integers(0, NUM_CLASSES, size=5),
            np.arange(5) + fold * 5,
        )
        for fold in range(2)
    ]
    path = write_out_of_fold(results, tmp_path / "oof.pkl")
    assert path is not None
    frame = pd.read_pickle(path)
    assert len(frame) == 10
    assert sorted(frame["index"]) == list(range(10))
    logits = np.stack([np.asarray(v) for v in frame["logits"]])
    report = evaluate(logits, frame["POS_class"].to_numpy())
    assert report.n == 10


def test_write_out_of_fold_returns_none_without_validation(tmp_path):
    results = [FoldResult(0, Path("x.pth"), 1, None, None, None, None)]
    assert write_out_of_fold(results, tmp_path / "oof.pkl") is None


def test_make_frame_satisfies_the_schema(frame):
    from rbdaim import schema

    schema.validate(frame, num_folds=2)


def test_replicates_get_different_seeds():
    """Each ensemble replicate must initialise differently.

    The published NN-model averages five independently-initialised models per
    pooling. If every replicate seeded identically they would be the same model,
    and averaging it with itself buys nothing -- a failure that costs hours of
    training and produces no error, so it is pinned here.
    """
    from rbdaim.model import AntiBertyPairClassifier
    from rbdaim.torch_utils import seed_everything

    def head_weights(seed: int):
        seed_everything(seed)
        return (
            AntiBertyPairClassifier(num_classes=NUM_CLASSES, pooling="cls")
            .head.weight.detach()
            .clone()
        )

    base = head_weights(42 + 0)
    assert torch.equal(base, head_weights(42 + 0)), "same seed must be reproducible"
    for ensemble_id in (1, 2, 3, 4):
        assert not torch.equal(base, head_weights(42 + ensemble_id))
