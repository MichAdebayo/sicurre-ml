from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.evaluation.hub_onnx import HubOnnxPredictor, HubTransformersPredictor


class FakeApi:
    def __init__(self, resolved: str) -> None:
        self.resolved = resolved

    def model_info(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(sha=self.resolved)


def test_onnx_predictor_requires_immutable_revision(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: FakeApi("different"), snapshot_download=lambda **_: None),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace()),
    )

    with pytest.raises(ValueError, match="must be immutable"):
        HubOnnxPredictor(
            repo_id="owner/model",
            revision="a" * 40,
            token=None,
            cache_dir=tmp_path,
        )


def test_onnx_predictor_downloads_and_predicts(monkeypatch, tmp_path: Path) -> None:
    revision = "a" * 40

    class FakeSession:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

        def run(self, *_: object) -> list[np.ndarray]:
            return [np.array([[0.1, 0.8, 0.2]])]

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(_: str) -> object:
            return lambda *args, **kwargs: {
                "input_ids": np.array([[1, 2]]),
                "attention_mask": np.array([[1, 1]]),
            }

    def fake_download(**kwargs: object) -> None:
        cache = Path(str(kwargs["local_dir"]))
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "model.onnx").write_bytes(b"model")
        (cache / "config.json").write_text(
            json.dumps({"id2label": {"0": "phishing", "1": "spam", "2": "legitimate"}})
        )

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: FakeApi(revision), snapshot_download=fake_download),
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FakeSession),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeTokenizer),
    )

    predictor = HubOnnxPredictor(
        repo_id="owner/model",
        revision=revision,
        token=None,
        cache_dir=tmp_path,
    )

    assert predictor.predict("message") == "spam"


def test_transformers_predictor_predicts_with_pinned_revision(monkeypatch) -> None:
    revision = "b" * 40

    class FakeTensor:
        def __getitem__(self, _: int) -> "FakeTensor":
            return self

    class FakeModel:
        config = SimpleNamespace(id2label={0: "PHISHING", 1: "LEGITIMATE"})

        def eval(self) -> None:
            pass

        def __call__(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(logits=FakeTensor())

    class TokenizerFactory:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            assert args[0] == "owner/model"
            assert kwargs["revision"] == revision
            return lambda *a, **k: {"input_ids": object()}

    class ModelFactory:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeModel:
            assert args[0] == "owner/model"
            assert kwargs["revision"] == revision
            return FakeModel()

    class NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_: object) -> None:
            return None

    fake_torch = SimpleNamespace(
        no_grad=NoGrad,
        argmax=lambda _: SimpleNamespace(item=lambda: 1),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: FakeApi(revision)),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForSequenceClassification=ModelFactory,
            AutoTokenizer=TokenizerFactory,
        ),
    )

    predictor = HubTransformersPredictor(
        repo_id="owner/model", revision=revision, token=None
    )

    assert predictor.predict("message") == "legitimate"
