"""Run lineage: what data, what code, what settings produced an artifact.

Every training run stamps its checkpoints and metrics with these fields, so a
result on disk can be traced back to the exact inputs that made it. Content
hashes rather than paths, because a path says nothing about what was in it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import rbdaim


@dataclass(frozen=True)
class Lineage:
    """Provenance stamp for one run."""

    package_version: str
    data_sha256: str
    data_bytes: int
    config_sha256: str
    git_commit: str
    python_version: str
    torch_version: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return the stamp as a JSON-serializable dict."""
        return asdict(self)


def file_sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> tuple[str, int]:
    """Return the SHA-256 hex digest and byte size of a file."""
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _git_commit() -> str:
    """Return the current git commit, or ``"unknown"`` outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def build(data_path: str | Path, config_repr: str, seed: int) -> Lineage:
    """Build a :class:`Lineage` stamp for a run over ``data_path``."""
    import torch

    data_hash, data_size = file_sha256(data_path)
    return Lineage(
        package_version=rbdaim.__version__,
        data_sha256=data_hash,
        data_bytes=data_size,
        config_sha256=hashlib.sha256(config_repr.encode()).hexdigest(),
        git_commit=_git_commit(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        seed=seed,
    )


def write(lineage: Lineage, path: str | Path) -> None:
    """Write a lineage stamp to ``path`` as JSON."""
    Path(path).write_text(json.dumps(lineage.to_dict(), indent=2) + "\n")
