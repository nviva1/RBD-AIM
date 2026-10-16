"""Fold-aware checkpoint selection.

These tests exist because the first version of this module filtered only on the
``_full`` suffix and happily averaged in other folds' checkpoints, reporting
0.96-1.00 "held-out" accuracy on every fold but 0.
"""

from __future__ import annotations

import pytest

from rbdaim.scoring import NoHonestCheckpointsError, select_checkpoints
from rbdaim.trainer import parse_checkpoint_name


def make_weights(tmp_path, names):
    """Create empty files with the given checkpoint names."""
    for name in names:
        (tmp_path / name).touch()
    return tmp_path


# The actual published release: numbered replicates for fold 0 only.
PUBLISHED = [
    *(f"NN_cv_0_CLS_{i}.pth" for i in range(5)),
    *(f"NN_cv_0_CDR_{i}.pth" for i in range(5)),
    *(f"NN_cv_{k}_CLS_full.pth" for k in range(5)),
    *(f"NN_cv_{k}_CDR_full.pth" for k in range(5)),
]


class TestParseCheckpointName:
    def test_numbered_checkpoint_records_its_held_out_fold(self):
        parsed = parse_checkpoint_name("NN_cv_3_CLS_2.pth")
        assert parsed is not None
        assert parsed.held_out_fold == 3
        assert parsed.pooling == "cls"
        assert parsed.replicate == "2"

    def test_ens_infix_spelling_is_accepted(self):
        parsed = parse_checkpoint_name("NN_cv_3_CDR_ens_2.pth")
        assert parsed is not None
        assert parsed.held_out_fold == 3
        assert parsed.pooling == "cdr"

    def test_full_model_holds_out_nothing(self):
        parsed = parse_checkpoint_name("NN_cv_2_CLS_full.pth")
        assert parsed is not None
        assert parsed.held_out_fold is None
        assert not any(parsed.is_honest_for(f) for f in range(10))

    @pytest.mark.parametrize(
        "name",
        ["model.pth", "NN_cv_x_CLS_0.pth", "NN_cv_0_MLP_0.pth", "NN_cv_0_CLS_0.pt"],
    )
    def test_unrecognised_names_return_none(self, name):
        assert parse_checkpoint_name(name) is None

    def test_held_out_fold_is_honest_only_for_itself(self):
        parsed = parse_checkpoint_name("NN_cv_4_CLS_0.pth")
        assert parsed is not None
        assert parsed.is_honest_for(4)
        assert not parsed.is_honest_for(0)


class TestSelectCheckpoints:
    def test_keeps_only_checkpoints_that_held_out_the_fold(self, tmp_path):
        d = make_weights(
            tmp_path,
            ["NN_cv_0_CLS_0.pth", "NN_cv_1_CLS_0.pth", "NN_cv_0_CLS_full.pth"],
        )
        got = {p.name for p in select_checkpoints(d, fold=0, pooling="cls")}
        assert got == {"NN_cv_0_CLS_0.pth"}

    def test_other_folds_checkpoints_are_excluded(self, tmp_path):
        """The bug: NN_cv_0_* trained on fold 3's rows, so it cannot score fold 3."""
        d = make_weights(tmp_path, PUBLISHED)
        with pytest.raises(NoHonestCheckpointsError, match="held out fold 3"):
            select_checkpoints(d, fold=3, pooling="cls")

    def test_the_error_names_the_folds_that_are_scorable(self, tmp_path):
        d = make_weights(tmp_path, PUBLISHED)
        with pytest.raises(NoHonestCheckpointsError) as excinfo:
            select_checkpoints(d, fold=7, pooling="cdr")
        assert "[0]" in str(excinfo.value)

    def test_published_release_scores_fold_zero_with_five_replicates(self, tmp_path):
        d = make_weights(tmp_path, PUBLISHED)
        kept = select_checkpoints(d, fold=0, pooling="cls")
        assert len(kept) == 5
        assert all(p.name.startswith("NN_cv_0_CLS_") for p in kept)
        assert not any(p.name.endswith("_full.pth") for p in kept)

    def test_pooling_is_respected(self, tmp_path):
        d = make_weights(tmp_path, PUBLISHED)
        assert all(
            "CDR" in p.name for p in select_checkpoints(d, fold=0, pooling="cdr")
        )

    def test_full_models_included_only_when_explicitly_allowed(self, tmp_path):
        d = make_weights(tmp_path, PUBLISHED)
        kept = select_checkpoints(d, fold=0, pooling="cls", allow_full_models=True)
        assert len(kept) == 10
        assert sum(p.name.endswith("_full.pth") for p in kept) == 5

    def test_other_folds_included_only_when_explicitly_allowed(self, tmp_path):
        d = make_weights(
            tmp_path, ["NN_cv_0_CLS_0.pth", "NN_cv_1_CLS_0.pth", "NN_cv_2_CLS_0.pth"]
        )
        kept = select_checkpoints(d, fold=0, pooling="cls", allow_other_folds=True)
        assert len(kept) == 3

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(NoHonestCheckpointsError, match="none"):
            select_checkpoints(tmp_path, fold=0, pooling="cls")

    def test_only_full_models_available_is_an_error_not_a_silent_leak(self, tmp_path):
        d = make_weights(tmp_path, ["NN_cv_0_CLS_full.pth", "NN_cv_1_CLS_full.pth"])
        with pytest.raises(NoHonestCheckpointsError):
            select_checkpoints(d, fold=0, pooling="cls")
