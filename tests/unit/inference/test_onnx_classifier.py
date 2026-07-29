from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.inference import onnx_classifier


def test_hub_tag_is_resolved_then_snapshot_is_pinned(monkeypatch, tmp_path: Path) -> None:
    resolved_sha = "c" * 40
    calls: dict[str, str] = {}

    class FakeApi:
        def repo_info(self, _: str, **kwargs: str) -> SimpleNamespace:
            calls["requested_revision"] = kwargs["revision"]
            return SimpleNamespace(sha=resolved_sha)

    def fake_snapshot_download(**kwargs: str) -> None:
        calls["download_revision"] = kwargs["revision"]
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True)
        (local_dir / "model.onnx").write_bytes(b"onnx")
        (local_dir / "config.json").write_text("{}")
        (local_dir / "tokenizer.json").write_text("{}")

    monkeypatch.setenv("HF_USERNAME", "owner")
    monkeypatch.setenv("REPO_NAME", "model")
    monkeypatch.setenv("HF_MODEL_REVISION", "production")
    monkeypatch.setenv("ONNX_MODEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, snapshot_download=fake_snapshot_download),
    )

    revision_dir = tmp_path / "revisions" / resolved_sha
    assert onnx_classifier._pull_from_hub() == revision_dir
    assert calls == {
        "requested_revision": "production",
        "download_revision": resolved_sha,
    }
    assert (tmp_path / "active_sha.txt").read_text() == resolved_sha
    assert (revision_dir / "sha.txt").read_text() == resolved_sha


def test_incomplete_revision_cannot_reuse_flat_stale_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    resolved_sha = "d" * 40
    (tmp_path / "model.onnx").write_bytes(b"stale")

    class FakeApi:
        def repo_info(self, _: str, **__: str) -> SimpleNamespace:
            return SimpleNamespace(sha=resolved_sha)

    def fake_snapshot_download(**kwargs: str) -> None:
        Path(kwargs["local_dir"]).mkdir(parents=True)

    monkeypatch.setenv("HF_USERNAME", "owner")
    monkeypatch.setenv("REPO_NAME", "model")
    monkeypatch.setenv("HF_MODEL_REVISION", "production")
    monkeypatch.setenv("ONNX_MODEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, snapshot_download=fake_snapshot_download),
    )

    with pytest.raises(RuntimeError, match="lacks a complete deployable ONNX bundle"):
        onnx_classifier._pull_from_hub()

    assert not (tmp_path / "active_sha.txt").exists()
