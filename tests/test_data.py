"""Dataset construction, label encoding and the CDR offset contract."""

from __future__ import annotations

from rbdaim.constants import NUM_CLASSES, UNLABELED
from rbdaim.data import collate_fab, labeled_mask

from .conftest import make_frame


def test_labeled_mask_excludes_the_unlabeled_sentinel():
    frame = make_frame(n=20, unlabeled=4)
    mask = labeled_mask(frame)
    assert (~mask).sum() == 4
    assert (frame.loc[~mask, "POS_class"] == UNLABELED).all()


def test_dataset_encodes_an_unlabeled_row_as_all_zero():
    """Show why unlabeled rows must be filtered, using the real dataset class.

    BCE reads an all-zero target as "negative for all 12 classes", i.e. an
    antibody that binds nothing at all.
    """
    from rbdaim.data import EpitopePairDataset

    frame = make_frame(n=6, n_folds=2, unlabeled=2)
    ds = EpitopePairDataset(frame)
    labels = ds.labels
    unlabeled_rows = frame["POS_class"].to_numpy() == UNLABELED
    assert unlabeled_rows.sum() == 2
    assert labels[unlabeled_rows].sum() == 0.0
    assert (labels[~unlabeled_rows].sum(axis=1) == 1.0).all()
    # label_index keeps the sentinel so callers can filter on it.
    assert (ds.label_index[unlabeled_rows] == UNLABELED).all()


def test_collate_stacks_fixed_tensors_and_keeps_cdr_lists():
    import torch

    def example(i: int, n_cdr: int):
        return {
            "antibody_id": i,
            "input_ids_light": torch.zeros(8, dtype=torch.long),
            "attention_mask_light": torch.ones(8, dtype=torch.long),
            "input_ids_heavy": torch.zeros(8, dtype=torch.long),
            "attention_mask_heavy": torch.ones(8, dtype=torch.long),
            "cdr_ids_light": torch.arange(n_cdr),
            "cdr_ids_heavy": torch.arange(n_cdr + 1),
            "labels": torch.zeros(NUM_CLASSES),
        }

    batch = collate_fab([example(0, 3), example(1, 5)])
    assert batch["input_ids_light"].shape == (2, 8)
    # Variable-length CDR indices must stay per-sample, not be padded/stacked.
    assert [len(t) for t in batch["cdr_ids_light"]] == [3, 5]
    assert batch["antibody_id"].tolist() == [0, 1]
