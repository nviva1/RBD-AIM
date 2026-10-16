"""Download the released checkpoints from the Hugging Face Hub.

The published weights are not redistributed with this code. This module fetches
them on demand and hands back a local directory that :mod:`rbdaim.scoring` can
score from directly, so running the real model is one command rather than a
manual download.
"""

from __future__ import annotations

import os
from pathlib import Path

from rbdaim.logging import get_logger

log = get_logger(__name__)

DEFAULT_REPO_ID = "nviva1/rbd-aim-epitope"

#: Checkpoint filenames are the unit of download; nothing else in the repo is
#: needed to score, and pulling the README/attributes wastes a request each.
_CHECKPOINT_GLOB = "*.pth"


class WeightsUnavailableError(RuntimeError):
    """Raised when the released checkpoints could not be fetched."""


def download_weights(
    repo_id: str = DEFAULT_REPO_ID,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Fetch the released checkpoints and return the local directory holding them.

    Args:
        repo_id: Hugging Face model repo to pull from.
        revision: Branch, tag or commit SHA. ``None`` tracks the repo default,
            which means the bytes you score can change under you; pin it when a
            number needs to be reproducible.
        cache_dir: Override the Hugging Face cache location.

    Returns:
        Path to a directory of ``.pth`` checkpoints, named as the release names
        them (``NN_cv_{fold}_{CLS,CDR}_{replicate|full}.pth``).

    Raises:
        WeightsUnavailableError: if ``huggingface_hub`` is missing, or the
            download fails (offline, private repo, bad revision).

    """
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
        raise WeightsUnavailableError(
            "huggingface-hub is required to download the released checkpoints:\n"
            "    pip install huggingface-hub\n"
            "Alternatively download them by hand and pass --checkpoints <dir>."
        ) from exc

    if revision is None:
        log.info("weights_revision_unpinned", repo_id=repo_id)

    log.info("weights_download_start", repo_id=repo_id, revision=revision)
    try:
        local = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            allow_patterns=[_CHECKPOINT_GLOB],
        )
    except Exception as exc:  # hub raises a wide family of errors
        raise WeightsUnavailableError(
            f"could not download checkpoints from {repo_id!r}: {exc}\n"
            "Check network access, or download the weights manually and pass "
            "--checkpoints <dir>. Set HF_TOKEN if the repo is gated."
        ) from exc

    directory = Path(local)
    found = sorted(p.name for p in directory.glob(_CHECKPOINT_GLOB))
    if not found:
        raise WeightsUnavailableError(
            f"{repo_id!r} contained no {_CHECKPOINT_GLOB} checkpoints "
            f"(revision {revision or 'default'})."
        )
    log.info(
        "weights_ready", path=str(directory), n_checkpoints=len(found), files=found
    )
    return directory


def resolve_checkpoint_dir(
    checkpoints: str | Path | None,
    *,
    from_hub: bool,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
) -> Path:
    """Resolve a checkpoint directory from either a local path or the Hub.

    Raises:
        ValueError: if neither source was requested, or both were.

    """
    if from_hub and checkpoints is not None:
        raise ValueError(
            "pass either --checkpoints <dir> or --from-hub, not both: they name "
            "two different sets of weights and silently preferring one would "
            "make the reported numbers ambiguous."
        )
    if from_hub:
        return download_weights(
            repo_id, revision=revision, cache_dir=os.environ.get("RBDAIM_CACHE")
        )
    if checkpoints is None:
        raise ValueError("no weights specified: pass --checkpoints <dir> or --from-hub")
    path = Path(checkpoints).expanduser()
    if not path.is_dir():
        raise ValueError(f"checkpoint directory not found: {path}")
    return path
