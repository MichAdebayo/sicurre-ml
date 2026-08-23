from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def test_pinned_complete_revision_uses_cache_without_hub(monkeypatch, tmp_path: Path) -> None:
    revision = "e" * 40
    revision_dir = tmp_path / "revisions" / revision
    revision_dir.mkdir(parents=True)
    for name in ("model.onnx", "config.json", "tokenizer.json"):
        (revision_dir / name).write_bytes(b"complete")
    monkeypatch.setenv("HF_USERNAME", "owner")
    monkeypatch.setenv("REPO_NAME", "model")
    monkeypatch.setenv("HF_MODEL_REVISION", revision)
    monkeypatch.setenv("ONNX_MODEL_CACHE_DIR", str(tmp_path))

    assert onnx_classifier._pull_from_hub() == revision_dir
    assert (tmp_path / "active_sha.txt").read_text() == revision


def test_hub_revision_must_resolve_to_sha(monkeypatch, tmp_path: Path) -> None:
    class FakeApi:
        def repo_info(self, *_: object, **__: object) -> SimpleNamespace:
            return SimpleNamespace(sha=None)

    monkeypatch.setenv("HF_USERNAME", "owner")
    monkeypatch.setenv("REPO_NAME", "model")
    monkeypatch.setenv("ONNX_MODEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, snapshot_download=lambda **_: None),
    )

    with pytest.raises(RuntimeError, match="did not resolve"):
        onnx_classifier._pull_from_hub()


def test_classify_onnx_uses_session_inputs_and_softmax(monkeypatch) -> None:
    class FakeTokenizer:
        def __call__(self, *_: object, **__: object) -> dict[str, np.ndarray]:
            return {
                "input_ids": np.array([[1, 2]], dtype=np.int32),
                "attention_mask": np.array([[1, 1]], dtype=np.int32),
                "token_type_ids": np.array([[0, 0]], dtype=np.int32),
            }

    class FakeSession:
        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="token_type_ids")]

        def run(self, _: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert inputs["input_ids"].dtype == np.int64
            assert inputs["token_type_ids"].dtype == np.int64
            return [np.array([[0.0, 3.0, 1.0]])]

    monkeypatch.setattr(
        onnx_classifier,
        "_load_session_and_tokenizer",
        lambda: (
            FakeSession(),
            FakeTokenizer(),
            {0: "phishing", 1: "spam", 2: "legitimate"},
        ),
    )

    result = onnx_classifier.classify_onnx("message")

    assert result.label == "spam"
    assert result.confidence == pytest.approx(result.raw_scores["spam"])
    assert sum(result.raw_scores.values()) == pytest.approx(1.0)


def test_load_session_uses_configured_labels(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"model")
    (tmp_path / "config.json").write_text(
        json.dumps({"id2label": {"0": "phishing", "1": "spam", "2": "legitimate"}})
    )

    class FakeOptions:
        intra_op_num_threads = 0
        inter_op_num_threads = 0
        execution_mode = None
        graph_optimization_level = None

    fake_session = object()
    monkeypatch.setattr(onnx_classifier, "_pull_from_hub", lambda: tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=FakeOptions,
            ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
            InferenceSession=lambda *args, **kwargs: fake_session,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: "tokenizer")
        ),
    )
    onnx_classifier._load_session_and_tokenizer.cache_clear()

    session, tokenizer, labels = onnx_classifier._load_session_and_tokenizer()

    assert session is fake_session
    assert tokenizer == "tokenizer"
    assert labels[2] == "legitimate"
    onnx_classifier._load_session_and_tokenizer.cache_clear()


def test_model_version_prefers_active_sha(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ONNX_MODEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_SHA", "fallback")
    assert onnx_classifier.get_model_version() == "fallback"

    (tmp_path / "active_sha.txt").write_text("f" * 40)
    assert onnx_classifier.get_model_version() == "f" * 40
