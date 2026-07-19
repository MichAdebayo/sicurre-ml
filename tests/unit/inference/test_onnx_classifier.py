from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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
        (tmp_path / "model.onnx").write_bytes(b"onnx")

    monkeypatch.setenv("HF_USERNAME", "owner")
    monkeypatch.setenv("REPO_NAME", "model")
    monkeypatch.setenv("HF_MODEL_REVISION", "production")
    monkeypatch.setenv("ONNX_MODEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, snapshot_download=fake_snapshot_download),
    )

    assert onnx_classifier._pull_from_hub() == tmp_path
    assert calls == {
        "requested_revision": "production",
        "download_revision": resolved_sha,
    }
    assert (tmp_path / "sha.txt").read_text() == resolved_sha
