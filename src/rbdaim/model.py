"""The epitope classifier model.

The original code had two ~95%-identical classes (``AntiBERTyFAB_CLS_Pair`` and
``AntiBERTyFAB_CDR_Pair``) differing only in how per-chain embeddings were
pooled. This unifies them into one model parameterized by ``pooling``.

Batching: the two AntiBERTy towers run on a full ``(B, T)`` batch. CLS pooling
uses the pooled ``[CLS]`` output directly; CDR pooling mean-pools each sample's
variable-length CDR positions individually (cheap gather after the expensive
encoder forward). CDR indices therefore arrive as a *list* of per-sample 1-D
tensors, one per batch element.
"""

from __future__ import annotations

import os
from typing import TypedDict

import antiberty
import torch
import torch.nn.functional as F
from antiberty import AntiBERTy
from torch import nn

_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.realpath(antiberty.__file__)),
    "trained_models",
    "AntiBERTy_md_smooth",
)


class ModelOutput(TypedDict):
    """Forward-pass outputs."""

    logits: torch.Tensor
    embeddings: torch.Tensor
    loss: torch.Tensor | None


class AntiBertyPairClassifier(nn.Module):
    """Paired heavy/light AntiBERTy encoder with a linear epitope head.

    Each chain is encoded by its own AntiBERTy tower. Per-chain representations
    are pooled (``"cls"`` token embedding, or mean over CDR-residue embeddings),
    concatenated, and projected to per-epitope logits.

    Task shape: this is **single-label multi-class** classification — every
    labeled antibody has exactly one ``POS_class``. The published model trains it
    with per-class ``binary_cross_entropy_with_logits`` against the one-hot
    target (optionally class-weighted), not with softmax cross-entropy. That is
    preserved here for checkpoint compatibility, and it is why inference reports
    independent per-class sigmoid scores rather than a softmax distribution; see
    :mod:`rbdaim.predictor`.
    """

    pos_weight: torch.Tensor  # registered buffer; declared for the type checker

    def __init__(
        self, num_classes: int = 12, pooling: str = "cls", dropout: float = 0.0
    ):
        """Initialize the classifier.

        Args:
            num_classes: Number of epitope classes.
            pooling: ``"cls"`` or ``"cdr"``.
            dropout: Dropout applied to the pooled embedding before the head.

        """
        super().__init__()
        if pooling not in {"cls", "cdr"}:
            raise ValueError(f"pooling must be 'cls' or 'cdr', got {pooling!r}")
        self.pooling = pooling
        self.num_classes = num_classes
        self.encoder_light = AntiBERTy.from_pretrained(_CHECKPOINT_PATH)
        self.encoder_heavy = AntiBERTy.from_pretrained(_CHECKPOINT_PATH)
        hidden = self.encoder_light.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, num_classes)
        # Per-class positive weights for BCE (class imbalance); None => unweighted.
        self.register_buffer("pos_weight", torch.ones(num_classes))
        self._use_pos_weight = False

    def set_class_weights(self, pos_weight: torch.Tensor) -> None:
        """Enable class-weighted BCE with the given per-class positive weights."""
        self.pos_weight = pos_weight.to(self.pos_weight.device)
        self._use_pos_weight = True

    def forward(
        self,
        input_ids_light: torch.Tensor,
        attention_mask_light: torch.Tensor,
        cdr_ids_light: list[torch.Tensor],
        input_ids_heavy: torch.Tensor,
        attention_mask_heavy: torch.Tensor,
        cdr_ids_heavy: list[torch.Tensor],
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        """Run a forward pass on a batch; compute BCE loss when ``labels`` given."""
        light = self._pool(
            self.encoder_light, input_ids_light, attention_mask_light, cdr_ids_light
        )
        heavy = self._pool(
            self.encoder_heavy, input_ids_heavy, attention_mask_heavy, cdr_ids_heavy
        )
        embeddings = torch.cat([light, heavy], dim=-1)
        logits = self.head(self.dropout(embeddings))
        loss = self._loss(logits, labels) if labels is not None else None
        return ModelOutput(logits=logits, embeddings=embeddings, loss=loss)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pos_weight = self.pos_weight if self._use_pos_weight else None
        return F.binary_cross_entropy_with_logits(
            logits, labels.float(), pos_weight=pos_weight
        )

    def _pool(
        self,
        encoder: AntiBERTy,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cdr_ids: list[torch.Tensor],
    ) -> torch.Tensor:
        """Pool a batch of chains into ``(B, hidden)`` representations."""
        outputs = encoder.bert(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "cls":
            pooled: torch.Tensor = outputs[1]  # (B, hidden) pooled [CLS]
            return pooled
        # CDR pooling: per sample, mean over its CDR-residue token embeddings.
        token_embeddings = outputs[0]  # (B, T, hidden)
        per_sample = [
            token_embeddings[i].index_select(0, cdr_ids[i]).mean(dim=0)
            for i in range(token_embeddings.shape[0])
        ]
        return torch.stack(per_sample, dim=0)
