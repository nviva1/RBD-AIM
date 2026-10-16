"""Resolving checkpoints from a local directory or the Hugging Face Hub."""

from __future__ import annotations

import pytest

from rbdaim import weights


def _never_download(*args, **kwargs):
    """Stand-in for the downloader: reaching the network is a test failure."""
    raise AssertionError("the Hub downloader must not be reached in this test")


class TestResolveCheckpointDir:
    def test_local_directory_is_returned(self, tmp_path):
        assert weights.resolve_checkpoint_dir(tmp_path, from_hub=False) == tmp_path

    def test_missing_local_directory_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="checkpoint directory not found"):
            weights.resolve_checkpoint_dir(tmp_path / "absent", from_hub=False)

    def test_a_file_is_not_a_checkpoint_directory(self, tmp_path):
        f = tmp_path / "NN_cv_0_CLS_0.pth"
        f.write_bytes(b"")
        with pytest.raises(ValueError, match="checkpoint directory not found"):
            weights.resolve_checkpoint_dir(f, from_hub=False)

    def test_neither_source_is_an_error(self):
        with pytest.raises(ValueError, match="no weights specified"):
            weights.resolve_checkpoint_dir(None, from_hub=False)

    def test_both_sources_is_an_error(self, tmp_path, monkeypatch):
        """Silently preferring one would make the reported numbers ambiguous."""
        # Stub the downloader rather than relying on the guard to prevent it:
        # if the guard regresses, this must fail, not start a multi-GB download.
        monkeypatch.setattr(weights, "download_weights", _never_download, raising=True)
        with pytest.raises(ValueError, match="not both"):
            weights.resolve_checkpoint_dir(tmp_path, from_hub=True)

    def test_from_hub_delegates_to_the_downloader(self, tmp_path, monkeypatch):
        seen: dict[str, object] = {}

        def fake_download(repo_id, *, revision=None, cache_dir=None):
            seen.update(repo_id=repo_id, revision=revision)
            return tmp_path

        monkeypatch.setattr(weights, "download_weights", fake_download)
        got = weights.resolve_checkpoint_dir(
            None, from_hub=True, repo_id="acme/w", revision="abc123"
        )
        assert got == tmp_path
        assert seen == {"repo_id": "acme/w", "revision": "abc123"}


class TestDownloadWeights:
    def _patch_hub(self, monkeypatch, snapshot):
        """Install a fake ``huggingface_hub.snapshot_download``."""
        import sys
        import types

        module = types.ModuleType("huggingface_hub")
        module.snapshot_download = snapshot  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    def test_returns_the_snapshot_directory(self, tmp_path, monkeypatch):
        (tmp_path / "NN_cv_0_CLS_0.pth").write_bytes(b"")
        self._patch_hub(monkeypatch, lambda **kw: str(tmp_path))
        assert weights.download_weights("acme/w") == tmp_path

    def test_requests_only_checkpoints(self, tmp_path, monkeypatch):
        (tmp_path / "NN_cv_0_CLS_0.pth").write_bytes(b"")
        seen: dict[str, object] = {}

        def snapshot(**kw):
            seen.update(kw)
            return str(tmp_path)

        self._patch_hub(monkeypatch, snapshot)
        weights.download_weights("acme/w", revision="v1")
        assert seen["allow_patterns"] == ["*.pth"]
        assert seen["repo_id"] == "acme/w"
        assert seen["revision"] == "v1"

    def test_an_empty_snapshot_is_an_error(self, tmp_path, monkeypatch):
        """A repo with no .pth would otherwise fail later, inside scoring."""
        self._patch_hub(monkeypatch, lambda **kw: str(tmp_path))
        with pytest.raises(weights.WeightsUnavailableError, match=r"no .* checkpoints"):
            weights.download_weights("acme/w")

    def test_a_download_failure_names_the_repo_and_the_manual_route(self, monkeypatch):
        def boom(**kw):
            raise OSError("offline")

        self._patch_hub(monkeypatch, boom)
        with pytest.raises(weights.WeightsUnavailableError) as exc:
            weights.download_weights("acme/w")
        assert "acme/w" in str(exc.value)
        assert "--checkpoints" in str(exc.value)

    def test_a_missing_hub_package_is_a_typed_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                raise ModuleNotFoundError("No module named 'huggingface_hub'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(weights.WeightsUnavailableError, match="pip install"):
            weights.download_weights("acme/w")
