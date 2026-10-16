"""Score existing checkpoints over a dataset fold.

This is the "do these weights actually work" path: it loads real checkpoints,
runs them over a held-out fold of the real dataset, and reports metrics. It is
separate from :mod:`rbdaim.trainer` on purpose — evaluating must never retrain.

Checkpoint selection is **fold-aware**, and that is the whole point of this
module. A checkpoint may only score the fold it held out. Selecting on the
``_full`` suffix alone is not enough: ``NN_cv_3_CLS_ens_0.pth`` held out fold 3,
so scoring fold 1 with it is scoring training data just as surely as a ``full``
model would. The released weight set makes this concrete — every numbered
checkpoint in it is ``NN_cv_0_*``, so **fold 0 is the only honestly scorable
fold from the public release**, and asking for any other fold is an error rather
than a flattering number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from rbdaim.config import Config
from rbdaim.data import (
    EpitopePairDataset,
    check_cdr_indices_fit,
    collate_fab,
    labeled_mask,
    load_dataframe,
)
from rbdaim.evaluate import EvalReport, evaluate
from rbdaim.logging import get_logger
from rbdaim.predictor import load_checkpoint
from rbdaim.torch_utils import resolve_device
from rbdaim.trainer import parse_checkpoint_name, split_mask

log = get_logger(__name__)


class NoHonestCheckpointsError(FileNotFoundError):
    """Raised when no checkpoint can score the requested fold without leaking."""


def select_checkpoints(
    checkpoint_dir: str | Path,
    *,
    fold: int,
    pooling: str,
    allow_full_models: bool = False,
    allow_other_folds: bool = False,
) -> list[Path]:
    """Return the checkpoints that may honestly score ``fold``.

    Keeps only checkpoints whose filename says they held ``fold`` out. ``full``
    models and other folds' models are excluded unless explicitly allowed, and
    allowing either means the result is a training-set score.

    Raises:
        NoHonestCheckpointsError: if nothing in the directory qualifies. The
            message names the folds that *are* scorable there, because the usual
            cause is asking for a fold whose checkpoints were never released.

    """
    ckpt_dir = Path(checkpoint_dir).expanduser()
    candidates = sorted(ckpt_dir.glob("NN_cv_*.pth"))

    keep: list[Path] = []
    leaky_full = 0
    leaky_other: set[int] = set()
    scorable_folds: set[int] = set()

    for path in candidates:
        parsed = parse_checkpoint_name(path.name)
        if parsed is None or parsed.pooling != pooling.lower():
            continue
        if parsed.held_out_fold is None:
            leaky_full += 1
            if allow_full_models:
                keep.append(path)
            continue
        scorable_folds.add(parsed.held_out_fold)
        if parsed.is_honest_for(fold):
            keep.append(path)
        else:
            leaky_other.add(parsed.held_out_fold)
            if allow_other_folds:
                keep.append(path)

    if leaky_full or leaky_other:
        log.info(
            "excluded_leaking_checkpoints",
            fold=fold,
            n_full_models=leaky_full if not allow_full_models else 0,
            other_folds=sorted(leaky_other) if not allow_other_folds else [],
            reason="trained on rows in this fold's validation split",
        )
    if not keep:
        raise NoHonestCheckpointsError(
            f"no {pooling.upper()} checkpoint in {ckpt_dir} held out fold {fold}, "
            f"so none can score it without evaluating on training data. "
            f"Folds scorable here: {sorted(scorable_folds) or 'none'}. "
            "(The published release ships numbered checkpoints for fold 0 only; "
            "every other file is a full model that trained on everything.)"
        )
    return keep


@torch.no_grad()
def score_checkpoints(
    config: Config,
    *,
    checkpoint_dir: str | Path,
    fold: int = 0,
    pooling: str | None = None,
    ensemble: bool = True,
    allow_full_models: bool = False,
    allow_other_folds: bool = False,
    curated_only: bool = True,
    batch_size: int = 8,
) -> EvalReport:
    """Score checkpoints on the held-out rows of one cross-validation fold.

    Ensemble logits are averaged across checkpoints, matching how the published
    ensemble is applied.

    ``pooling`` defaults to ``config.model.pooling`` so that ``model=antiberty_cdr``
    and an explicit argument cannot disagree about which model is being run.

    ``curated_only`` restricts scoring to antibodies with a curated assignment
    (``good`` in {0, 1}). Setting it False scores every labeled row — a larger,
    noisier set whose numbers are not comparable to the curated ones.

    Raises:
        NoHonestCheckpointsError: if no checkpoint held out ``fold``.
        ValueError: if the fold holds no scorable rows.

    """
    pooling = pooling or config.model.pooling
    paths = select_checkpoints(
        checkpoint_dir,
        fold=fold,
        pooling=pooling,
        allow_full_models=allow_full_models,
        allow_other_folds=allow_other_folds,
    )
    if not ensemble:
        paths = paths[:1]

    device = resolve_device(config.train.device)
    df = load_dataframe(config.data.path, config.data.num_folds)
    is_val = split_mask(df[f"{config.data.split_prefix}_{fold}"])
    scorable = labeled_mask(df, config.data.label_column)
    if curated_only:
        scorable &= df["good"].isin([0, 1])
    eval_df = df[scorable & is_val]
    if eval_df.empty:
        raise ValueError(
            f"fold {fold} has no scorable rows "
            f"(column {config.data.split_prefix}_{fold}, "
            f"curated_only={curated_only})"
        )

    ds = EpitopePairDataset(eval_df, config.model.num_classes, config.data.max_length)
    check_cdr_indices_fit(ds, pooling=pooling, max_length=config.data.max_length)
    # shuffle=False: ds.label_index must stay row-aligned with the logits below.
    loader: DataLoader[Any] = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fab
    )
    log.info(
        "scoring",
        n_checkpoints=len(paths),
        checkpoints=[p.name for p in paths],
        n_rows=len(ds),
        fold=fold,
        pooling=pooling,
        leak_guard="fold-aware"
        if not (allow_full_models or allow_other_folds)
        else "DISABLED - scores include training data",
    )

    per_checkpoint: list[np.ndarray] = []
    for path in paths:
        model = load_checkpoint(
            path, pooling=pooling, device=device, num_classes=config.model.num_classes
        )
        batch_logits = []
        for batch in loader:
            inputs = {
                k: ([t.to(device) for t in v] if isinstance(v, list) else v.to(device))
                for k, v in batch.items()
                if k not in {"antibody_id", "labels"}
            }
            batch_logits.append(model(**inputs)["logits"].cpu().numpy())
        per_checkpoint.append(np.concatenate(batch_logits))
        del model

    logits = np.mean(per_checkpoint, axis=0)
    labels = ds.label_index
    if len(logits) != len(labels):
        raise RuntimeError(
            f"logit/label misalignment: {len(logits)} logits for {len(labels)} rows"
        )
    report = evaluate(logits, labels, seed=config.seed)
    log.info("scored", n=report.n, n_checkpoints=len(paths), **report.metrics)
    return report
