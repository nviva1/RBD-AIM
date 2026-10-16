"""Antibody numbering and CDR index extraction."""

from __future__ import annotations

import importlib.util

import pytest

from rbdaim.constants import HEAVY_CDR_POSITIONS
from rbdaim.numbering import (
    ChainNumberingError,
    _position_number,
    check_cdr_positions,
    number_chain,
    number_heavy,
    number_light,
)

# abnumber delegates numbering to ANARCI, which is bioconda-only. Skipping is
# honest here; failing would just report the environment, not the code.
anarci_required = pytest.mark.skipif(
    importlib.util.find_spec("anarci") is None,
    reason="ANARCI not installed (bioconda); see environment.yml",
)

# A real therapeutic Fab pair (trastuzumab variable domains).
HEAVY = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKG"
    "RFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
LIGHT = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSR"
    "SGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("H100", 100), ("H100A", 100), ("L24", 24), ("L95B", 95)],
)
def test_position_number_strips_prefix_and_insertion_code(raw, expected):
    assert _position_number(raw) == expected


@pytest.mark.slow
@anarci_required
def test_numbering_yields_one_based_cdr_indices_within_the_sequence():
    """CDR indices are 1-based ordinals into the gap-free numbered sequence."""
    for numbered in (number_heavy(HEAVY), number_light(LIGHT)):
        assert numbered.cdr_positions
        assert min(numbered.cdr_positions) >= 1
        assert max(numbered.cdr_positions) <= len(numbered.sequence)
        assert len(set(numbered.cdr_positions)) == len(numbered.cdr_positions)


@pytest.mark.slow
@anarci_required
def test_numbered_sequence_is_gap_free_and_from_the_input():
    numbered = number_heavy(HEAVY)
    assert "-" not in numbered.sequence
    assert set(numbered.sequence) <= set(HEAVY)


class TestNumberChainGuards:
    """Guards in `number_chain`, which needs no ANARCI to exercise.

    It only iterates (position, residue) pairs, so a plain list stands in for a
    Chain. These guards were previously untested: the only numbering tests
    needed a bioconda install and skipped everywhere, CI included.
    """

    def test_an_empty_chain_is_rejected(self):
        """Abnumber returns an empty Chain rather than raising.

        It does this for sequences it cannot align; unchecked, that mean-pools
        to nan and reports class A.
        """
        with pytest.raises(ChainNumberingError, match="empty chain"):
            number_chain([], HEAVY_CDR_POSITIONS)

    def test_a_chain_with_no_cdr_residues_is_rejected(self):
        framework_only = [("H1", "E"), ("H2", "V"), ("H3", "Q")]
        with pytest.raises(ChainNumberingError, match="no CDR residues"):
            number_chain(framework_only, HEAVY_CDR_POSITIONS)

    def test_cdr_ordinals_are_one_based_into_the_gap_free_sequence(self):
        """Ordinal counting must not be thrown off by non-CDR residues."""
        cdr = min(HEAVY_CDR_POSITIONS)
        chain = [("H1", "E"), ("H2", "V"), (f"H{cdr}", "W"), ("H2000", "K")]
        numbered = number_chain(chain, HEAVY_CDR_POSITIONS)
        assert numbered.sequence == "EVWK"
        assert numbered.cdr_positions == (3,)  # 1-based, counts framework too


class TestCheckCdrPositions:
    def test_empty_positions_are_rejected(self):
        with pytest.raises(ChainNumberingError, match="no CDR positions"):
            check_cdr_positions([], chain="light", n_residues=10)

    def test_a_position_past_the_sequence_is_rejected(self):
        with pytest.raises(ChainNumberingError, match="outside"):
            check_cdr_positions([1, 11], chain="heavy", n_residues=10)

    def test_a_zero_or_negative_position_is_rejected(self):
        """These are 1-based ordinals; 0 would silently shift to -1."""
        with pytest.raises(ChainNumberingError, match="outside"):
            check_cdr_positions([0, 5], chain="light", n_residues=10)

    def test_positions_spanning_the_whole_sequence_are_accepted(self):
        check_cdr_positions([1, 10], chain="light", n_residues=10)
