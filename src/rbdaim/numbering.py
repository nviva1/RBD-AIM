"""Antibody numbering and CDR extraction (Chothia scheme).

The original research code inlined this parsing loop in three separate places
with subtle copy-paste differences. It lives here once, typed and tested.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from abnumber import Chain

from rbdaim.constants import HEAVY_CDR_POSITIONS, LIGHT_CDR_POSITIONS


class AnarciMissingError(RuntimeError):
    """Raised when ANARCI, abnumber's numbering backend, is not installed."""


class ChainNumberingError(ValueError):
    """Raised when a sequence cannot be numbered as an antibody variable domain."""


@dataclass(frozen=True)
class NumberedChain:
    """A Chothia-numbered antibody chain.

    Attributes:
        sequence: Gap-free amino-acid sequence in numbering order.
        cdr_positions: 1-based indices (into ``sequence``) of CDR residues.

    """

    sequence: str
    cdr_positions: tuple[int, ...]


def _position_number(pos: object) -> int:
    """Extract the integer residue number from an abnumber position.

    abnumber positions stringify as e.g. ``"H100"`` or ``"H100A"`` (insertion
    codes). We drop the chain-type prefix and any trailing insertion letter.
    """
    text = str(pos)[1:]  # strip leading 'H'/'L'
    if text and not text[-1].isdigit():
        text = text[:-1]  # strip insertion code
    return int(text)


def number_chain(chain: Chain, cdr_positions: frozenset[int]) -> NumberedChain:
    """Return ``chain`` numbered, with its CDR residue indices recorded.

    Args:
        chain: An abnumber ``Chain`` (already Chothia-numbered).
        cdr_positions: Chothia residue numbers considered CDR for this chain.

    Returns:
        The gap-free sequence and the 1-based indices of its CDR residues.

    Raises:
        ChainNumberingError: if the chain numbers to nothing, or yields no CDR
            residues. abnumber does **not** raise for every unusable input — an
            antibody-like sequence it cannot align (e.g. one carrying a long
            CDR-H3 insertion) comes back as a silently *empty* ``Chain``. Left
            unchecked, that empty chain mean-pools a zero-length index to
            ``nan`` and ``argmax`` then reports class "A" with ``nan`` scores,
            which looks exactly like a real prediction.

    """
    sequence: list[str] = []
    cdr_index: list[int] = []
    for ordinal, (pos, aa) in enumerate(chain, start=1):
        if _position_number(pos) in cdr_positions:
            cdr_index.append(ordinal)
        sequence.append(aa)
    if not sequence:
        raise ChainNumberingError(
            "antibody numbering produced an empty chain: the sequence could not "
            "be aligned to a variable domain. This happens for sequences that "
            "look antibody-like but carry an unalignable region (e.g. a long "
            "CDR-H3 insertion); abnumber returns an empty chain rather than "
            "raising."
        )
    if not cdr_index:
        raise ChainNumberingError(
            f"antibody numbering produced no CDR residues for a {len(sequence)}-"
            "residue chain. The chain may be a fragment, or numbered against the "
            "wrong chain type (heavy vs light)."
        )
    return NumberedChain("".join(sequence), tuple(cdr_index))


def check_cdr_positions(
    positions: Sequence[int] | np.ndarray, *, chain: str, n_residues: int
) -> None:
    """Validate 1-based CDR ordinals supplied from outside :func:`number_chain`.

    :func:`number_chain` guards the sequences it numbers itself, but rows that
    arrive already numbered (the dataset contract, and
    :meth:`~rbdaim.predictor.EpitopePredictor.predict_numbered`) carry their CDR
    ordinals as data and bypass it entirely.

    Two failure modes are worth the check. An **empty** array mean-pools a
    zero-length gather to ``nan``, and ``argmax`` of an all-``nan`` vector
    returns 0 — a confident-looking prediction of the first class. An
    **out-of-range** ordinal silently gathers a padding token instead of a
    residue, which is wrong without being obviously wrong.

    Args:
        positions: 1-based CDR residue ordinals.
        chain: ``"heavy"`` or ``"light"``, for the error message.
        n_residues: Length of the numbered sequence the ordinals index into.

    Raises:
        ChainNumberingError: if ``positions`` is empty or leaves ``[1, n]``.

    """
    if len(positions) == 0:
        raise ChainNumberingError(
            f"no CDR positions for the {chain} chain. Mean-pooling an empty CDR "
            "index yields nan, and argmax of an all-nan score vector silently "
            f"returns the first epitope class. Check '{chain}_ids_cdr'."
        )
    lo, hi = int(min(positions)), int(max(positions))
    if lo < 1 or hi > n_residues:
        raise ChainNumberingError(
            f"{chain} CDR positions span [{lo}, {hi}], outside the "
            f"{n_residues}-residue numbered sequence. These are 1-based "
            f"ordinals into '{chain}_anarci'; an out-of-range one gathers a "
            "padding token rather than a residue."
        )


def number_light(sequence: str) -> NumberedChain:
    """Chothia-number a light-chain sequence string."""
    return number_chain(_chain(sequence), LIGHT_CDR_POSITIONS)


def number_heavy(sequence: str) -> NumberedChain:
    """Chothia-number a heavy-chain sequence string."""
    return number_chain(_chain(sequence), HEAVY_CDR_POSITIONS)


def _chain(sequence: str) -> Chain:
    """Build a Chothia-numbered abnumber ``Chain`` from a raw sequence.

    Raises:
        AnarciMissingError: if ANARCI is not installed. abnumber delegates the
            actual numbering to ANARCI, which is distributed through bioconda
            rather than PyPI, so ``pip install`` alone leaves this import
            missing and the failure otherwise surfaces as a bare
            ``ModuleNotFoundError`` from deep inside abnumber.

    """
    try:
        return Chain(sequence.replace("-", ""), scheme="chothia")
    except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
        if exc.name != "anarci":
            raise
        raise AnarciMissingError(
            "ANARCI is required to number raw antibody sequences, and it is "
            "not installed. Two routes:\n"
            "    pip install 'rbdaim-classifier[predict]'   # + HMMER, see below\n"
            "    conda env create -f environment.yml        # brings both\n"
            "The PyPI `anarci` is only a wrapper: it shells out to `hmmscan` "
            "from HMMER, a C program that pip cannot install. Without HMMER on "
            "PATH you get FileNotFoundError: 'hmmscan', not this message.\n"
            "Training and scoring need none of this — the dataset ships "
            "pre-numbered sequences; only `rbdaim predict` on raw ones does."
        ) from exc
