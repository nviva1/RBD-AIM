"""Run provenance stamping."""

from __future__ import annotations

import json

from rbdaim import lineage


def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "data.bin"
    payload = b"epitope" * 1000
    path.write_bytes(payload)
    digest, size = lineage.file_sha256(path)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert size == len(payload)


def test_hash_distinguishes_content_not_filename(tmp_path):
    """Lineage must key on content: a renamed file is the same data."""
    a, b, c = (tmp_path / n for n in ("a.pkl", "b.pkl", "c.pkl"))
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    c.write_bytes(b"different")
    assert lineage.file_sha256(a)[0] == lineage.file_sha256(b)[0]
    assert lineage.file_sha256(a)[0] != lineage.file_sha256(c)[0]


def test_build_stamps_data_config_and_versions(tmp_path):
    path = tmp_path / "data.pkl"
    path.write_bytes(b"rows")
    stamp = lineage.build(path, "Config(seed=42)", seed=42)
    assert stamp.data_sha256 == lineage.file_sha256(path)[0]
    assert stamp.data_bytes == 4
    assert stamp.seed == 42
    assert stamp.torch_version and stamp.python_version and stamp.package_version


def test_config_hash_changes_with_the_config(tmp_path):
    path = tmp_path / "data.pkl"
    path.write_bytes(b"rows")
    a = lineage.build(path, "Config(epochs=4)", seed=0)
    b = lineage.build(path, "Config(epochs=8)", seed=0)
    assert a.config_sha256 != b.config_sha256
    assert a.data_sha256 == b.data_sha256


def test_write_round_trips_as_json(tmp_path):
    path = tmp_path / "data.pkl"
    path.write_bytes(b"rows")
    stamp = lineage.build(path, "cfg", seed=1)
    out = tmp_path / "lineage.json"
    lineage.write(stamp, out)
    assert json.loads(out.read_text()) == stamp.to_dict()
