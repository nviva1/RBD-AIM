"""Inference: load one or an ensemble of checkpoints and predict epitopes.

Encapsulates numbering + tokenizing a Fab pair, running the model(s), averaging
logits across the ensemble, and reporting per-epitope scores.

On scores: the model is trained with per-class BCE, so each output is an
independent log-odds. We report ``sigmoid(logit)`` per class — these do not sum
to 1, and that is correct rather than a bug. They are uncalibrated: treat them
as a ranking, not as P(epitope).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rbdaim.constants import EPITOPE_CLASSES, NUM_CLASSES
from rbdaim.data import EpitopePairDataset, collate_fab
from rbdaim.evaluate import class_scores
from rbdaim.logging import get_logger
from rbdaim.model import AntiBertyPairClassifier
from rbdaim.numbering import (
    AnarciMissingError,
    check_cdr_positions,
    number_heavy,
    number_light,
)
from rbdaim.torch_utils import resolve_device

log = get_logger(__name__)

# The published RBD-AIM checkpoints use the original module names; this refactor
# renamed them. The architecture is identical, so a prefix remap loads them 1:1.
_LEGACY_PREFIX_MAP = {
    "model_l.": "encoder_light.",
    "model_h.": "encoder_heavy.",
    "epitopes.": "head.",
}


def _remap_legacy(state: dict[str, Any]) -> dict[str, Any]:
    """Remap legacy paper-checkpoint keys to this package's module names."""
    if not any(k.startswith("model_l.") for k in state):
        return state
    remapped = {}
    for key, value in state.items():
        for old, new in _LEGACY_PREFIX_MAP.items():
            if key.startswith(old):
                key = new + key[len(old) :]
                break
        remapped[key] = value
    return remapped


def load_checkpoint(
    path: str | Path,
    *,
    pooling: str = "cls",
    device: torch.device | None = None,
    num_classes: int = NUM_CLASSES,
) -> AntiBertyPairClassifier:
    """Load one checkpoint into a model, verifying every weight arrived.

    Handles both checkpoint shapes — a rich dict (``{"model_state": ...}``) and
    a bare ``state_dict`` — and remaps the published checkpoints' legacy module
    names.

    The load is verified. ``load_state_dict(strict=False)`` without checking its
    result is how a model full of randomly-initialized weights goes on to report
    plausible-looking accuracy: every key silently fails to match, nothing
    raises, and the numbers are of an untrained network.

    Raises:
        RuntimeError: if any weight is missing or unexpected.

    """
    dev = device or resolve_device("auto")
    model = AntiBertyPairClassifier(num_classes=num_classes, pooling=pooling)
    loaded = torch.load(path, map_location=dev, weights_only=False)
    if isinstance(loaded, dict) and "model_state" in loaded:
        loaded = loaded["model_state"]
    state = _remap_legacy(loaded)
    result = model.load_state_dict(state, strict=False)
    # pos_weight is a class-imbalance buffer used only during training; the
    # published checkpoints predate it, so its absence is expected and harmless.
    # Every weight that affects a prediction must still be present.
    unresolved = set(result.missing_keys) - {"pos_weight"}
    if unresolved or result.unexpected_keys:
        raise RuntimeError(
            f"incompatible checkpoint {Path(path).name}: "
            f"missing={sorted(unresolved)[:5]} "
            f"unexpected={list(result.unexpected_keys)[:5]}"
        )
    return model.to(dev).eval()


@dataclass(frozen=True)
class Prediction:
    """A single epitope prediction.

    Attributes:
        predicted_class: Epitope class with the highest score.
        scores: Per-class sigmoid scores. Independent per class, so they do not
            sum to 1, and they are not calibrated probabilities.
        embedding_norm: Mean L2 norm of the pooled embedding across the
            ensemble. Useful as a crude out-of-distribution signal.

    """

    predicted_class: str
    scores: dict[str, float]
    embedding_norm: float

    def ranked(self) -> list[tuple[str, float]]:
        """Return (class, score) pairs sorted by descending score."""
        return sorted(self.scores.items(), key=lambda kv: -kv[1])


class EpitopePredictor:
    """Loads checkpoint(s) and predicts epitope class for Fab pairs."""

    def __init__(
        self,
        models: list[AntiBertyPairClassifier],
        device: torch.device,
        max_length: int = 150,
    ):
        """Wrap already-loaded models; prefer :meth:`from_dir`."""
        self._models = models
        self._device = device
        self._max_length = max_length

    @property
    def n_models(self) -> int:
        """Number of checkpoints in the ensemble."""
        return len(self._models)

    @classmethod
    def from_dir(
        cls,
        model_dir: str | Path,
        *,
        pooling: str = "cls",
        ensemble: bool = True,
        device: str = "auto",
        num_classes: int = NUM_CLASSES,
        max_length: int = 150,
    ) -> EpitopePredictor:
        """Load checkpoints matching the pooling from ``model_dir``.

        Loads every ``NN_cv_*_{POOL}_*.pth`` when ``ensemble`` is true, else the
        single ``NN_cv_0_{POOL}_full.pth`` (falling back to any one match).

        Raises:
            FileNotFoundError: if the directory holds no matching checkpoint.

        """
        model_dir = Path(model_dir).expanduser()
        pool = pooling.upper()
        candidates = sorted(model_dir.glob(f"NN_cv_*_{pool}_*.pth"))
        if not ensemble:
            preferred = [p for p in candidates if p.name.endswith("_full.pth")]
            candidates = preferred[:1] or candidates[:1]
        if not candidates:
            raise FileNotFoundError(
                f"no {pool} checkpoints in {model_dir} "
                f"(looking for NN_cv_*_{pool}_*.pth). "
                "See 'Getting the weights' in the README."
            )

        dev = resolve_device(device)
        models = [
            load_checkpoint(c, pooling=pooling, device=dev, num_classes=num_classes)
            for c in candidates
        ]
        log.info(
            "predictor_loaded", n_models=len(models), pooling=pooling, device=str(dev)
        )
        return cls(models, dev, max_length=max_length)

    @torch.no_grad()
    def predict(self, heavy: str, light: str) -> Prediction:
        """Predict the epitope class for one Fab (heavy + light sequence).

        Raises:
            ValueError: if either sequence cannot be antibody-numbered.

        """
        return self._run(self._encode(heavy, light))

    @torch.no_grad()
    def predict_numbered(self, row: pd.Series) -> Prediction:
        """Predict from a row that is already antibody-numbered.

        Takes a row of the dataset contract (``heavy_anarci``, ``light_anarci``,
        ``heavy_ids_cdr``, ``light_ids_cdr``) and skips numbering entirely. Use
        this for dataset rows: it is faster, and it needs no ANARCI install.

        Skipping numbering also skips its guards, so the CDR ordinals carried by
        the row are validated here instead — see
        :func:`rbdaim.numbering.check_cdr_positions`.

        Raises:
            ChainNumberingError: if either chain's CDR ordinals are empty or
                fall outside its numbered sequence.

        """
        light = np.asarray(row["light_ids_cdr"], dtype=np.int64)
        heavy = np.asarray(row["heavy_ids_cdr"], dtype=np.int64)
        check_cdr_positions(light, chain="light", n_residues=len(row["light_anarci"]))
        check_cdr_positions(heavy, chain="heavy", n_residues=len(row["heavy_anarci"]))
        frame = pd.DataFrame(
            {
                "index": [0],
                "light_anarci": [row["light_anarci"]],
                "heavy_anarci": [row["heavy_anarci"]],
                "light_ids_cdr": [light],
                "heavy_ids_cdr": [heavy],
                "POS_class": [-1],
            }
        )
        return self._run(self._batch_inputs(frame))

    def _run(self, item: dict[str, Any]) -> Prediction:
        """Run the ensemble over one prepared model input.

        Raises:
            ValueError: if the ensemble produced a non-finite logit.

        """
        logits = []
        embedding_norm = 0.0
        for model in self._models:
            out = model(**item, labels=None)
            logits.append(out["logits"].cpu().numpy().reshape(1, -1))
            embedding_norm += float(out["embeddings"].norm().cpu())
        mean_logits = np.mean(logits, axis=0)
        # argmax does not skip nan — it *prefers* it, so a single non-finite
        # class hijacks the prediction and the rest of the row is never
        # consulted. Refusing to return is the only safe answer here.
        if not np.isfinite(mean_logits).all():
            n_bad = int((~np.isfinite(mean_logits)).sum())
            raise ValueError(
                f"model produced {n_bad} non-finite logit(s) of "
                f"{mean_logits.size}; refusing to report a prediction. "
                "np.argmax prefers nan, so this would otherwise surface as a "
                "confident-looking class with nan scores. Usual cause: an empty "
                "or out-of-range CDR index mean-pooling to nan."
            )
        scores = class_scores(mean_logits).reshape(-1)
        return Prediction(
            predicted_class=EPITOPE_CLASSES[int(np.argmax(mean_logits.reshape(-1)))],
            scores={c: float(s) for c, s in zip(EPITOPE_CLASSES, scores, strict=True)},
            embedding_norm=embedding_norm / len(self._models),
        )

    def _encode(self, heavy: str, light: str) -> dict[str, Any]:
        """Return a Fab pair numbered and tokenized as a one-example batch."""
        try:
            nl = number_light(light)
            nh = number_heavy(heavy)
        except AnarciMissingError:
            raise  # a missing backend is not a malformed input; say which it is
        except Exception as exc:  # abnumber raises several distinct types
            raise ValueError(
                f"could not antibody-number the input as a heavy/light Fab pair: {exc}"
            ) from exc
        frame = pd.DataFrame(
            {
                "index": [0],
                "light_anarci": [nl.sequence],
                "heavy_anarci": [nh.sequence],
                "light_ids_cdr": [np.asarray(nl.cdr_positions, dtype=np.int64)],
                "heavy_ids_cdr": [np.asarray(nh.cdr_positions, dtype=np.int64)],
                "POS_class": [-1],
            }
        )
        return self._batch_inputs(frame)

    def _batch_inputs(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Tokenize a one-row numbered frame into model forward kwargs."""
        ds = EpitopePairDataset(frame, max_length=self._max_length)
        batch = collate_fab([ds[0]])
        model_args = (
            "input_ids_light",
            "attention_mask_light",
            "cdr_ids_light",
            "input_ids_heavy",
            "attention_mask_heavy",
            "cdr_ids_heavy",
        )
        moved: dict[str, Any] = {}
        for key in model_args:
            value = batch[key]  # type: ignore[literal-required]
            moved[key] = (
                [t.to(self._device) for t in value]
                if isinstance(value, list)
                else value.to(self._device)
            )
        return moved
