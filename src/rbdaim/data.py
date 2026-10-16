"""Dataset and tokenization for paired-chain epitope classification.

Refactor of the original ``EpitopesPairDatasetCDR``: same tensors and semantics,
but typed, documented, and decoupled from disk loading (loading lives in
:func:`load_dataframe`, which validates the contract).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import TypedDict

import antiberty
import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset

from rbdaim import schema
from rbdaim.constants import NUM_CLASSES, UNLABELED

# AntiBERTy ships its tokenizer vocab alongside the model checkpoint.
_ANTIBERTY_DIR = os.path.dirname(os.path.realpath(antiberty.__file__))
_VOCAB_FILE = os.path.join(_ANTIBERTY_DIR, "trained_models", "vocab.txt")


@functools.lru_cache(maxsize=1)
def tokenizer() -> transformers.BertTokenizer:
    """Return the shared AntiBERTy tokenizer.

    Cached: constructing it reads ``vocab.txt`` from disk, and inference used to
    pay that cost on every single prediction.
    """
    tok: transformers.BertTokenizer = transformers.BertTokenizer(  # type: ignore[no-untyped-call]
        vocab_file=_VOCAB_FILE, do_lower_case=False
    )
    return tok


class FabExample(TypedDict):
    """One tokenized Fab pair (a single dataset item)."""

    antibody_id: int
    input_ids_light: torch.Tensor
    attention_mask_light: torch.Tensor
    input_ids_heavy: torch.Tensor
    attention_mask_heavy: torch.Tensor
    cdr_ids_light: torch.Tensor
    cdr_ids_heavy: torch.Tensor
    labels: torch.Tensor


class FabBatch(TypedDict):
    """A collated batch of Fab examples.

    Fixed-width tensors are stacked; variable-length CDR index tensors stay as
    per-sample lists (the model gathers them per sample).
    """

    antibody_id: torch.Tensor
    input_ids_light: torch.Tensor
    attention_mask_light: torch.Tensor
    input_ids_heavy: torch.Tensor
    attention_mask_heavy: torch.Tensor
    cdr_ids_light: list[torch.Tensor]
    cdr_ids_heavy: list[torch.Tensor]
    labels: torch.Tensor


def collate_fab(batch: list[FabExample]) -> FabBatch:
    """Collate Fab examples into a batch (keeps CDR indices as per-sample lists)."""
    return FabBatch(
        antibody_id=torch.tensor([b["antibody_id"] for b in batch], dtype=torch.long),
        input_ids_light=torch.stack([b["input_ids_light"] for b in batch]),
        attention_mask_light=torch.stack([b["attention_mask_light"] for b in batch]),
        input_ids_heavy=torch.stack([b["input_ids_heavy"] for b in batch]),
        attention_mask_heavy=torch.stack([b["attention_mask_heavy"] for b in batch]),
        cdr_ids_light=[b["cdr_ids_light"] for b in batch],
        cdr_ids_heavy=[b["cdr_ids_heavy"] for b in batch],
        labels=torch.stack([b["labels"] for b in batch]),
    )


def load_dataframe(path: str | Path, num_folds: int = 10) -> pd.DataFrame:
    """Load and validate the Fab dataframe from a local path.

    Args:
        path: Filesystem path to the pickled dataframe.
        num_folds: Number of cross-validation split columns expected.

    Returns:
        The validated dataframe with a reset index.

    Raises:
        FileNotFoundError: with a pointer to the README, if ``path`` is absent.

    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"dataset not found: {path}\n"
            "The labeled Fab dataset is not distributed with this code. "
            "See 'Getting the data' in the README for the download, then pass "
            "it explicitly: rbdaim train data.path=/path/to/dataset_production.pkl"
        )
    df = pd.read_pickle(path).reset_index(drop=True)
    return schema.validate(df, num_folds=num_folds)


def labeled_mask(df: pd.DataFrame, label_column: str = "POS_class") -> pd.Series:
    """Boolean mask of rows carrying a real epitope label.

    The production frame mixes labeled antibodies with unlabeled ones
    (``POS_class == -1``). An unlabeled row one-hot-encodes to an all-zero
    target, which BCE reads as "negative for all 12 classes" — i.e. training on
    it teaches the model that a real antibody binds nothing. Unlabeled rows are
    excluded from training and from scoring; this mask is how.
    """
    return df[label_column].astype(int) != UNLABELED


class EpitopePairDataset(Dataset[FabExample]):
    """Tokenized paired heavy/light Fab sequences with epitope labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        num_classes: int = NUM_CLASSES,
        max_length: int = 150,
    ):
        """Initialize the dataset.

        Args:
            df: A (subset of the) validated Fab dataframe.
            num_classes: Number of epitope classes (one-hot label width).
            max_length: Tokenizer padding/truncation length.

        """
        self._num_classes = num_classes
        self._max_length = max_length
        self._tokenizer = tokenizer()

        self._ids: list[int] = df["index"].to_list()
        self._light_seq: list[str] = df["light_anarci"].to_list()
        self._heavy_seq: list[str] = df["heavy_anarci"].to_list()

        # 1-based residue ordinals, shifted by -1. This does not line up with
        # the tokenizer: `_encode` adds special tokens, so residue r sits at
        # token index r and CDR pooling reads a window one residue left. The
        # offset matches the released checkpoints' indexing -- changing it
        # invalidates them.
        self._light_cdr = [
            np.asarray(a, dtype=np.int64) - 1 for a in df["light_ids_cdr"]
        ]
        self._heavy_cdr = [
            np.asarray(a, dtype=np.int64) - 1 for a in df["heavy_ids_cdr"]
        ]

        # One-hot targets. POS_class == -1 (unlabeled) yields an all-zero row;
        # callers are expected to have filtered those out with `labeled_mask`.
        labels = df["POS_class"].to_numpy()
        self._labels = np.where(
            labels[:, None] == np.arange(num_classes)[None, :], 1.0, 0.0
        ).astype(np.float32)
        self._label_index = labels.astype(np.int64)

    @property
    def labels(self) -> np.ndarray:
        """One-hot label matrix ``(n, num_classes)`` for the whole dataset."""
        return self._labels

    @property
    def label_index(self) -> np.ndarray:
        """Integer class per row ``(n,)``; ``-1`` where unlabeled."""
        return self._label_index

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> FabExample:
        light = self._encode(self._light_seq[idx])
        heavy = self._encode(self._heavy_seq[idx])
        return FabExample(
            antibody_id=self._ids[idx],
            input_ids_light=light["input_ids"].flatten(),
            attention_mask_light=light["attention_mask"].flatten(),
            input_ids_heavy=heavy["input_ids"].flatten(),
            attention_mask_heavy=heavy["attention_mask"].flatten(),
            cdr_ids_light=torch.as_tensor(self._light_cdr[idx], dtype=torch.long),
            cdr_ids_heavy=torch.as_tensor(self._heavy_cdr[idx], dtype=torch.long),
            labels=torch.as_tensor(self._labels[idx], dtype=torch.float32),
        )

    def _encode(self, sequence: str) -> transformers.BatchEncoding:
        """Tokenize a residue sequence (residues space-separated for AntiBERTy)."""
        encoded: transformers.BatchEncoding = self._tokenizer.encode_plus(
            " ".join(sequence),
            add_special_tokens=True,
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return encoded


def check_cdr_indices_fit(
    dataset: EpitopePairDataset, *, pooling: str, max_length: int
) -> None:
    """Verify every CDR index addresses a real token, for CDR pooling.

    CDR ordinals index into the tokenized sequence, which the tokenizer
    truncates to ``max_length``. If a chain is longer than that, its tail CDR
    residues no longer exist as tokens and ``index_select`` raises a bare
    ``IndexError`` from inside the encoder. Checking up front turns that into a
    message naming the setting to change.

    No-op for ``cls`` pooling, which does not gather CDR positions — but note
    that truncation still silently shortens the sequence there, so the same
    ``max_length`` advice applies.

    Raises:
        ValueError: if any CDR index falls outside the tokenized length.

    """
    if pooling != "cdr":
        return
    # Token layout is [CLS] r1 ... rN [SEP], so the last addressable token index
    # is max_length - 1.
    limit = max_length - 1
    worst = max(
        (int(a.max()) for a in (*dataset._light_cdr, *dataset._heavy_cdr) if a.size),
        default=-1,
    )
    if worst > limit:
        raise ValueError(
            f"CDR index {worst} exceeds the tokenized length "
            f"(data.max_length={max_length}, last token index {limit}). "
            f"Raise data.max_length to at least {worst + 2} for cdr pooling."
        )
