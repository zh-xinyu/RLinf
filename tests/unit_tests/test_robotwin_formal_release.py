# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Verify immutable release identity before any large resource transfer."""

import hashlib
import json

import pytest

from toolkits.robotwin_mopd.formal import download_resources as release


def setup_release(tmp_path, monkeypatch):
    here = tmp_path / "code"
    here.mkdir()
    record = {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    (here / "resources.json").write_text(
        json.dumps({"files": {"student/model": record}})
    )
    remote = tmp_path / "remote.json"
    remote.write_text(
        json.dumps(
            {"files": {"student/model": dict(record, path_in_repo="remote/model")}}
        )
    )
    source = tmp_path / "download"
    source.write_bytes(b"abc")
    calls = []

    def fetch(repository, path, revision):
        assert repository == release.REPOSITORY and revision == release.REVISION
        calls.append(path)
        return str(remote if path == release.MANIFEST else source)

    monkeypatch.setattr(release, "HERE", here)
    monkeypatch.setattr(release, "hf_hub_download", fetch)
    return remote, source, calls


def test_download_reuses_verified_local_and_rejects_corruption(tmp_path, monkeypatch):
    _, source, calls = setup_release(tmp_path, monkeypatch)
    root = tmp_path / "resources"
    release.download(root)
    assert (root / "student/model").resolve() == source
    calls.clear()
    release.download(root)
    assert calls == [release.MANIFEST]
    source.write_bytes(b"bad")
    with pytest.raises(ValueError, match="identity mismatch"):
        release.download(root)


def test_wrong_release_fails_before_downloading_weights(tmp_path, monkeypatch):
    remote, _, calls = setup_release(tmp_path, monkeypatch)
    record = json.loads(remote.read_text())
    record["files"]["student/model"]["sha256"] = "wrong"
    remote.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checked-out recipe"):
        release.download(tmp_path / "resources")
    assert calls == [release.MANIFEST]
