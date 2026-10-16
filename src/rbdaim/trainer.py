"""Training loop and cross-validation runner.

* real batching (collate keeps variable-length CDR indices as per-sample lists),
* AdamW + linear warmup/decay schedule,
* optional class-weighted BCE for label imbalance,
* per-epoch validation (loss + macro-F1) on cross-validation folds,
* early stopping on the monitored metric with best-checkpoint selection,
* resume-from-checkpoint (optimizer + scheduler + epoch) after an interruption.

Two training modes, deliberately kept distinct:

``full_model=false`` (default)
    Cross-validation. Fold *k* holds out ``cross_val_test_k`` for validation and
    trains on the rest. These runs produce the out-of-fold predictions you
    evaluate and report.

``full_model=true``
    The shipped inference model: trains on every non-holdout row. There is no
    held-out split left, so this mode runs a **fixed** number of epochs with no
    early stopping and reports **no** validation metrics. Scoring a model on
    data it trained on is not a measurement, so this mode refuses to produce
    one.

Checkpoints are rich dicts (``model_state`` + metadata); the shipped best model
is saved under the published inference naming
(``NN_cv_{fold}_{POOL}_{full|ens_n}.pth``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from rbdaim import lineage
from rbdaim.config import Config
from rbdaim.data import (
    EpitopePairDataset,
    FabBatch,
    collate_fab,
    labeled_mask,
    load_dataframe,
)
from rbdaim.logging import get_logger
from rbdaim.metrics import class_metrics
from rbdaim.model import AntiBertyPairClassifier
from rbdaim.torch_utils import resolve_device, seed_everything

log = get_logger(__name__)


@dataclass
class FoldResult:
    """Per-fold training output.

    For cross-validation folds the validation fields describe the best epoch.
    For ``full_model`` runs there is no validation split, so ``best_epoch`` is
    the final epoch and the validation fields are ``None``.
    """

    fold: int
    checkpoint_path: Path
    best_epoch: int
    best_val_f1_macro: float | None
    val_predictions: np.ndarray | None  # (n_val, num_classes) logits
    val_labels: np.ndarray | None  # (n_val,) integer class
    val_ids: np.ndarray | None  # (n_val,) antibody ids

    @property
    def has_validation(self) -> bool:
        """True when this fold produced honest held-out predictions."""
        return self.val_predictions is not None


def checkpoint_name(fold: int, pooling: str, *, full: bool, ensemble_id: int) -> str:
    """Checkpoint filename for a fold (the published inference naming scheme).

    ``full`` models (trained on everything but the holdout) are what inference
    loads by default: ``NN_cv_{fold}_{POOL}_full.pth``.
    """
    pool = pooling.upper()  # CLS / CDR (matches the original checkpoint scheme)
    suffix = "full" if full else f"ens_{ensemble_id}"
    return f"NN_cv_{fold}_{pool}_{suffix}.pth"


# NN_cv_{fold}_{POOL}_{full|ens_N|N}.pth. The published release drops the `ens_`
# infix and writes a bare replicate index, so both spellings are accepted.
_CHECKPOINT_RE = re.compile(
    r"^NN_cv_(?P<fold>\d+)_(?P<pool>CLS|CDR)_(?P<suffix>.+)\.pth$"
)


@dataclass(frozen=True)
class CheckpointName:
    """What a checkpoint filename says about how that model was trained.

    Attributes:
        held_out_fold: The cross-validation fold this model did **not** train
            on, or ``None`` for a ``full`` model, which held out nothing beyond
            the final holdout.
        pooling: ``"cls"`` or ``"cdr"``.
        replicate: Ensemble replicate label, or ``"full"``.

    """

    held_out_fold: int | None
    pooling: str
    replicate: str

    def is_honest_for(self, fold: int) -> bool:
        """Return True if scoring ``fold`` with this model avoids training data."""
        return self.held_out_fold == fold


def parse_checkpoint_name(name: str) -> CheckpointName | None:
    """Parse a checkpoint filename, or return ``None`` if it does not match.

    This is what makes fold-aware checkpoint selection possible: the filename is
    the only record of which rows a released checkpoint was allowed to see.
    """
    match = _CHECKPOINT_RE.match(name)
    if match is None:
        return None
    suffix = match.group("suffix")
    full = suffix == "full"
    return CheckpointName(
        held_out_fold=None if full else int(match.group("fold")),
        pooling=match.group("pool").lower(),
        replicate=suffix,
    )


def split_mask(series: pd.Series) -> pd.Series:
    """Boolean membership mask for a split column.

    Null entries (rows outside the cross-validation set) are treated as
    ``False`` — not in this split.
    """
    return series.fillna(False).infer_objects().astype(bool)


def class_pos_weights(labels_onehot: np.ndarray) -> torch.Tensor:
    """Per-class BCE positive weights (neg/pos) from one-hot training labels.

    Classes absent from the split get weight 1.0; rare classes get up-weighted.
    """
    pos = labels_onehot.sum(axis=0)
    neg = labels_onehot.shape[0] - pos
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def _loader(ds: EpitopePairDataset, cfg: Config, *, shuffle: bool) -> DataLoader[Any]:
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fab,
        num_workers=cfg.train.num_workers,
    )


def _model_inputs(batch: FabBatch, device: torch.device) -> dict[str, Any]:
    """Move a batch to ``device`` as model forward kwargs (drops ``antibody_id``)."""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "antibody_id":
            continue  # not a model input; used only for bookkeeping
        if isinstance(value, list):
            moved[key] = [t.to(device) for t in value]
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def _evaluate(
    model: AntiBertyPairClassifier, loader: DataLoader[Any], device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Score a loader: return (logits, int labels, ids, mean loss)."""
    model.eval()
    logits_all, labels_all, ids_all, losses = [], [], [], []
    for batch in loader:
        ids_all.append(batch["antibody_id"].numpy())
        labels_all.append(batch["labels"].argmax(dim=1).numpy())
        out = model(**_model_inputs(batch, device))
        logits_all.append(out["logits"].cpu().numpy())
        if out["loss"] is not None:
            losses.append(float(out["loss"].item()))
    return (
        np.concatenate(logits_all),
        np.concatenate(labels_all),
        np.concatenate(ids_all),
        float(np.mean(losses)) if losses else float("nan"),
    )


def _save_resume(
    path: Path,
    model: AntiBertyPairClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best: float,
    best_state: dict[str, Any] | None,
    best_eval: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    best_epoch: int,
    patience: int | None,
) -> None:
    """Persist everything needed to continue this fold as if uninterrupted.

    That means the whole selection state, not just the weights: the best model,
    the epoch it came from, the remaining patience budget, and the held-out
    predictions at that epoch. Dropping any of them makes a resumed fold differ
    from an uninterrupted one — losing ``best_eval`` in particular makes the
    fold return no predictions and vanish from the cross-validation average.
    """
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "epoch": epoch,
            "best_val_f1_macro": best,
            "best_state": best_state,
            "best_eval": best_eval,
            "best_epoch": best_epoch,
            "patience": patience,
        },
        path,
    )


def train_fold(
    df: pd.DataFrame, fold: int, config: Config, device: torch.device
) -> FoldResult:
    """Train one fold. See the module docstring for the two training modes."""
    full = config.train.full_model
    labeled = labeled_mask(df, config.data.label_column)
    in_fold = split_mask(df[f"{config.data.split_prefix}_{fold}"])
    holdout = split_mask(df[config.data.holdout_column])

    # The published split (`train_model.py:165-173`): train on every row not in
    # this fold, predict on the fold. There is no third validation split — the
    # fold IS the held-out set, and its predictions are the out-of-fold output.
    # `full_model` instead trains on everything except the final holdout.
    keep = ~holdout if full else ~in_fold
    if config.train.exclude_unlabeled:
        keep &= labeled
    if config.train.exclude_holdout_from_cv and not full:
        keep &= ~holdout
    train_df = df[keep]
    # Evaluation needs labels to score; unlabeled rows in the fold are dropped
    # from the metric only, not from training.
    val_df = df.iloc[0:0] if full else df[in_fold & labeled]

    log.info(
        "fold_split",
        fold=fold,
        mode="full" if full else "cv",
        n_train=len(train_df),
        n_val=len(val_df),
        n_unlabeled_in_train=int((~labeled & keep).sum()),
        n_holdout_in_train=int((holdout & keep).sum()),
    )

    train_ds = EpitopePairDataset(
        train_df, config.model.num_classes, config.data.max_length
    )
    train_loader = _loader(train_ds, config, shuffle=True)
    val_loader = (
        _loader(
            EpitopePairDataset(
                val_df, config.model.num_classes, config.data.max_length
            ),
            config,
            shuffle=False,
        )
        if len(val_df)
        else None
    )

    model = AntiBertyPairClassifier(
        num_classes=config.model.num_classes,
        pooling=config.model.pooling,
        dropout=config.model.dropout,
    ).to(device)
    if config.train.class_weighting:
        model.set_class_weights(class_pos_weights(train_ds.labels).to(device))

    optimizer = AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    # warmup_ratio 0.0 means no schedule at all: the original runs a constant
    # 2e-5 (its OneCycleLR is commented out at train_classifier.py:126). A
    # linear-decay schedule with zero warmup is NOT the same thing — it still
    # decays to zero — so the schedule is skipped entirely rather than zeroed.
    scheduler: Any = None
    if config.train.warmup_ratio > 0:
        total_steps = max(1, len(train_loader) * config.train.epochs)
        scheduler = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call]
            optimizer,
            num_warmup_steps=int(config.train.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )

    ckpt_dir = Path(config.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / checkpoint_name(
        fold,
        config.model.pooling,
        full=full,
        ensemble_id=config.train.ensemble_id,
    )
    # The resume file identifies the exact run: a different ensemble replicate,
    # or full vs cross-validation, trains on a different row set and must never
    # resume from this one. Sharing `last_cv{fold}_{POOL}.pt` across replicates
    # made every ensemble member resume from the first and come out identical.
    run_tag = "full" if full else f"ens_{config.train.ensemble_id}"
    resume_path = (
        ckpt_dir / f"last_cv{fold}_{config.model.pooling.upper()}_{run_tag}.pt"
    )

    start_epoch, best_f1, patience = 0, -1.0, config.train.patience
    best_state: dict[str, Any] | None = None
    best_eval: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    best_epoch = 0
    if config.train.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt["scheduler_state"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt["best_val_f1_macro"]
        best_state = ckpt.get("best_state")
        best_epoch = ckpt.get("best_epoch", 0)
        patience = ckpt.get("patience", config.train.patience)
        # Held-out predictions ride along too. Without them, a resumed fold that
        # never beats its stored high-water mark returns no predictions and
        # silently drops out of the cross-validation average.
        best_eval = ckpt.get("best_eval")
        log.info(
            "resumed",
            fold=fold,
            from_epoch=start_epoch,
            best_f1=best_f1,
            has_best_predictions=best_eval is not None,
        )
        if start_epoch >= config.train.epochs:
            raise ValueError(
                f"fold {fold} already completed {start_epoch} of "
                f"{config.train.epochs} epochs; there is nothing to resume. "
                f"Raise train.epochs, or delete {resume_path} to retrain from "
                "scratch."
            )

    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        train_losses: list[float] = []
        for batch in tqdm(
            train_loader, desc=f"fold {fold} epoch {epoch + 1}", leave=False
        ):
            out = model(**_model_inputs(batch, device))
            loss = out["loss"]
            assert loss is not None
            optimizer.zero_grad()
            loss.backward()
            if config.train.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.train.max_grad_norm
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        current_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

        if val_loader is None:
            # full_model: no held-out data, so no metric to early-stop on. The
            # last epoch is the shipped model, by definition.
            log.info("epoch_done", fold=fold, epoch=epoch + 1, train_loss=train_loss)
            best_state, best_epoch = current_state, epoch + 1
            _save_resume(
                resume_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_f1,
                best_state,
                best_eval,
                best_epoch,
                patience,
            )
            continue

        logits, labels, ids, val_loss = _evaluate(model, val_loader, device)
        val_f1 = class_metrics(labels, logits.argmax(axis=1)).f1_macro
        log.info(
            "epoch_done",
            fold=fold,
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            val_f1_macro=val_f1,
        )

        if config.train.patience is None:
            # Published behaviour: no model selection. Every epoch runs and the
            # final one is the model, so these predictions are genuinely
            # out-of-fold — nothing was chosen by looking at them.
            best_f1, best_epoch = val_f1, epoch + 1
            best_state, best_eval = current_state, (logits, labels, ids)
            _save_resume(
                resume_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_f1,
                best_state,
                best_eval,
                best_epoch,
                patience,
            )
            continue

        if val_f1 > best_f1 + config.train.min_delta:
            best_f1, best_epoch, patience = val_f1, epoch + 1, config.train.patience
            best_state = current_state
            best_eval = (logits, labels, ids)
            _save_resume(
                resume_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_f1,
                best_state,
                best_eval,
                best_epoch,
                patience,
            )
        else:
            # Unreachable when early stopping is off: that path continues above.
            patience = 0 if patience is None else patience - 1
            _save_resume(
                resume_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_f1,
                best_state,
                best_eval,
                best_epoch,
                patience,
            )
            if patience <= 0:
                log.info("early_stop", fold=fold, epoch=epoch + 1, best_f1=best_f1)
                break

    if best_state is None:
        raise RuntimeError(
            f"fold {fold} produced no checkpoint: train.epochs="
            f"{config.train.epochs}, start_epoch={start_epoch}. "
            "Nothing was trained — raise epochs or clear the resume file at "
            f"{resume_path}."
        )

    torch.save(
        {
            "model_state": best_state,
            "pooling": config.model.pooling,
            "num_classes": config.model.num_classes,
            "best_epoch": best_epoch,
            "best_val_f1_macro": None if val_loader is None else best_f1,
            "full_model": full,
        },
        best_path,
    )
    log.info("checkpoint_saved", path=str(best_path), best_epoch=best_epoch)
    # The fold is complete: drop the resume file so a rerun retrains rather than
    # "resuming" a finished run into an empty epoch loop.
    resume_path.unlink(missing_ok=True)

    if val_loader is None:
        return FoldResult(fold, best_path, best_epoch, None, None, None, None)
    if best_eval is None:
        raise RuntimeError(
            f"fold {fold} ran validation but kept no held-out predictions. "
            "This is a bug in the training loop, not a configuration problem."
        )
    logits, labels, ids = best_eval
    return FoldResult(fold, best_path, best_epoch, best_f1, logits, labels, ids)


def train(config: Config) -> list[FoldResult]:
    """Run cross-validation training; return per-fold results.

    Writes ``lineage.json`` and ``train_metrics.json`` into ``output_dir`` so a
    run's inputs and results are on disk next to its checkpoints.
    """
    # The seed must vary with the replicate. The published NN-model averages
    # five independently-initialised models per pooling; seeding every replicate
    # identically would make them the same model, and averaging it with itself
    # gains nothing. (The original sets no seed at all, so its replicates differ
    # by chance -- this makes that difference reproducible instead.)
    seed_everything(config.seed + config.train.ensemble_id)
    device = resolve_device(config.train.device)
    df = load_dataframe(config.data.path, config.data.num_folds)
    folds = config.train.folds or list(range(config.data.num_folds))

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lin = lineage.build(config.data.path, repr(config), config.seed)
    lineage.write(lin, out_dir / "lineage.json")

    log.info("training_start", device=str(device), folds=folds, **lin.to_dict())
    results = [train_fold(df, fold, config, device) for fold in folds]

    scored = [r for r in results if r.best_val_f1_macro is not None]
    summary: dict[str, Any] = {
        "folds": [
            {
                "fold": r.fold,
                "checkpoint": str(r.checkpoint_path),
                "best_epoch": r.best_epoch,
                "best_val_f1_macro": r.best_val_f1_macro,
            }
            for r in results
        ],
        "mean_val_f1_macro": (
            float(np.mean([r.best_val_f1_macro for r in scored])) if scored else None
        ),
        "lineage": lin.to_dict(),
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Persist the out-of-fold predictions themselves, not just their summary
    # scalars, so a run's held-out predictions survive for later analysis.
    oof_path = write_out_of_fold(results, out_dir / "out_of_fold.pkl")
    log.info(
        "training_done",
        mean_val_f1_macro=summary["mean_val_f1_macro"],
        out_of_fold=str(oof_path) if oof_path else None,
    )
    return results


def write_out_of_fold(results: list[FoldResult], path: Path) -> Path | None:
    """Write held-out logits, labels and ids to a pickle; ``None`` if there are none.

    One row per held-out antibody: its ``index``, its ``POS_class``, the fold
    that produced it, and a ``logits`` array. Curation flags are deliberately
    not carried here — they belong to the source dataset, and re-stating them
    from a training run would be inventing provenance. Join on ``index`` if you
    need them.
    """
    scored = [r for r in results if r.has_validation]
    if not scored:
        return None
    frame = pd.DataFrame(
        {
            "index": np.concatenate([r.val_ids for r in scored]),
            "POS_class": np.concatenate([r.val_labels for r in scored]),
            "logits": [row for r in scored for row in np.asarray(r.val_predictions)],
            "fold": np.concatenate(
                [np.full(len(r.val_labels), r.fold) for r in scored]  # type: ignore[arg-type]
            ),
        }
    )
    frame.to_pickle(path)
    return path


def out_of_fold(results: list[FoldResult]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate out-of-fold validation logits and labels across folds.

    Raises:
        ValueError: if no fold produced validation predictions (e.g. every run
            was ``full_model=true``, which has no held-out split to score).

    """
    scored = [r for r in results if r.has_validation]
    if not scored:
        raise ValueError(
            "no fold produced held-out predictions. full_model=true trains on "
            "everything, so there is nothing to evaluate — run cross-validation "
            "(train.full_model=false) to produce scorable out-of-fold outputs."
        )
    logits = np.concatenate([r.val_predictions for r in scored], axis=0)
    labels = np.concatenate([r.val_labels for r in scored], axis=0)
    return logits, labels
