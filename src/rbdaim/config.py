"""Typed configuration objects.

Hydra composes YAML into a nested ``DictConfig`` at runtime; these dataclasses
are the typed view the code actually consumes, so mypy checks config access and
IDEs autocomplete it.

Every field here is read by code. A setting that nothing consumes is a lie about
what the program does, so it does not get a dataclass field.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from rbdaim.constants import NUM_CLASSES


@dataclass(frozen=True)
class DataConfig:
    """Dataset location and the column contract.

    ``path`` has no default: the dataset is not redistributable with the code,
    and a plausible-looking default that only resolves on one machine is worse
    than an error. See the README for where to obtain it.
    """

    path: str
    max_length: int = 150
    label_column: str = "POS_class"
    split_prefix: str = "cross_val_test"
    holdout_column: str = "our_test"
    num_folds: int = 10


@dataclass(frozen=True)
class ModelConfig:
    """Classifier architecture options."""

    pooling: str = "cls"  # "cls" | "cdr"
    dropout: float = 0.0
    # Class count is a property of the label space, not a tunable. It lives in
    # rbdaim.constants; this mirror exists only so checkpoints record it.
    num_classes: int = NUM_CLASSES

    def __post_init__(self) -> None:
        """Validate enum-like fields at construction time."""
        if self.pooling not in {"cls", "cdr"}:
            raise ValueError(f"pooling must be 'cls' or 'cdr', got {self.pooling!r}")
        if self.num_classes != NUM_CLASSES:
            raise ValueError(
                f"num_classes must be {NUM_CLASSES} (the published label space), "
                f"got {self.num_classes}"
            )


@dataclass(frozen=True)
class TrainConfig:
    """Optimization and cross-validation settings."""

    # Defaults match the original research code. Every deviation is opt-in below.
    epochs: int = 4
    lr: float = 2e-5
    batch_size: int = 1
    # None disables clipping, matching the original's effective behaviour.
    max_grad_norm: float | None = None
    weight_decay: float = 0.0
    device: str = "auto"
    folds: list[int] | None = None
    # full_model: train on every non-holdout row (the shipped inference model).
    # There is then no held-out data, so no metrics are produced — see
    # rbdaim.trainer.train_fold.
    full_model: bool = False
    ensemble_id: int = 0
    checkpoint_dir: str = "weights"
    # 0.0 disables the LR schedule entirely, giving the original's constant LR.
    warmup_ratio: float = 0.0
    class_weighting: bool = False
    # --- Deviations from the original. All default to off. ---
    # Early stopping on held-out macro-F1, keeping the best epoch's weights
    # instead of the last. Note the kept epoch is chosen using the same rows it
    # is then scored on, so its held-out metric is selected-on and optimistic.
    patience: int | None = None
    min_delta: float = 0.0
    # Drop rows with no epitope label from training. Their one-hot is all-zero,
    # which BCE reads as "negative for all 12 classes" (22 rows).
    exclude_unlabeled: bool = False
    # Drop the final-holdout rows from cross-validation training. They are null
    # in every cross_val_test_k column, so they otherwise train in every fold.
    exclude_holdout_from_cv: bool = False
    num_workers: int = 0
    # Resume an interrupted fold from checkpoint_dir if a resume file is there.
    resume: bool = False

    def __post_init__(self) -> None:
        """Reject combinations the checkpoint naming scheme cannot express.

        Raises:
            ValueError: if ``full_model`` is combined with a non-zero
                ``ensemble_id``.

        """
        if self.full_model and self.ensemble_id != 0:
            raise ValueError(
                "train.full_model=true cannot be combined with "
                f"train.ensemble_id={self.ensemble_id}. A full model has no "
                "fold, so the published naming reuses the fold slot for the "
                "replicate: NN_cv_{0..4}_{POOL}_full.pth. ensemble_id has no "
                "slot of its own there, so honouring it would mean either "
                "inventing a filename the release does not use, or writing "
                "every replicate to NN_cv_{fold}_{POOL}_full.pth and silently "
                "overwriting the previous one. Vary train.folds to pick which "
                "replicate indices to write, or train.checkpoint_dir to keep "
                "separate sets apart."
            )


@dataclass(frozen=True)
class Config:
    """Top-level composed configuration."""

    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    seed: int = 42
    output_dir: str = "outputs"


def from_dictconfig(cfg: DictConfig) -> Config:
    """Convert a composed Hydra ``DictConfig`` into the typed :class:`Config`.

    Unknown keys raise. A silently-ignored key is how ``train.epocs=10`` runs a
    full training job with the default epoch count and nobody notices.

    Raises:
        ValueError: if a YAML section contains a key the dataclass does not
            declare, or if a required key is missing.

    """

    def _section(dc_cls: type, name: str) -> dict[str, Any]:
        raw = cfg.get(name)
        data: dict[str, Any] = dict(raw) if raw else {}
        allowed = {f.name for f in dataclasses.fields(dc_cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(
                f"unknown key(s) in config section {name!r}: {unknown}. "
                f"Known keys: {sorted(allowed)}"
            )
        return data

    known_top_level = {"data", "model", "train", "seed", "output_dir"}
    unknown_top = sorted(str(k) for k in cfg if str(k) not in known_top_level)
    if unknown_top:
        raise ValueError(
            f"unknown top-level config key(s): {unknown_top}. "
            f"Known keys: {sorted(known_top_level)}"
        )

    return Config(
        data=DataConfig(**_section(DataConfig, "data")),
        model=ModelConfig(**_section(ModelConfig, "model")),
        train=TrainConfig(**_section(TrainConfig, "train")),
        seed=int(cfg.get("seed", 42)),
        output_dir=str(cfg.get("output_dir", "outputs")),
    )
