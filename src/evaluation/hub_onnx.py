from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class HubOnnxPredictor:
    """ONNX predictor pinned to an immutable Hugging Face commit."""

    def __init__(
        self,
        *,
        repo_id: str,
        revision: str,
        token: str | None,
        cache_dir: Path,
    ) -> None:
        import onnxruntime as ort
        from huggingface_hub import HfApi, snapshot_download
        from transformers import AutoTokenizer

        resolved = HfApi().model_info(
            repo_id=repo_id,
            revision=revision,
            token=token,
        ).sha
        if not resolved or resolved != revision:
            raise ValueError(
                f"Model revision must be immutable: requested {revision}, resolved {resolved}"
            )
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
            local_dir=str(cache_dir),
            allow_patterns=["model.onnx", "config.json", "tokenizer*", "sentencepiece*"],
        )
        self._session = ort.InferenceSession(
            str(cache_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(str(cache_dir))
        config = json.loads((cache_dir / "config.json").read_text())
        self._id2label = {
            int(key): str(value).lower() for key, value in config["id2label"].items()
        }

    def predict(self, text: str) -> str:
        encoded: dict[str, Any] = self._tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=256,
            padding="max_length",
        )
        inputs = {
            item.name: encoded[item.name].astype(np.int64)
            for item in self._session.get_inputs()
            if item.name in encoded
        }
        logits = self._session.run(None, inputs)[0][0]
        return self._id2label[int(np.argmax(logits))]


class HubTransformersPredictor:
    """PyTorch evaluator pinned to an immutable Hugging Face commit."""

    def __init__(self, *, repo_id: str, revision: str, token: str | None) -> None:
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        resolved = HfApi().model_info(
            repo_id=repo_id,
            revision=revision,
            token=token,
        ).sha
        if not resolved or resolved != revision:
            raise ValueError(
                f"Model revision must be immutable: requested {revision}, resolved {resolved}"
            )
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            repo_id, revision=revision, token=token
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            repo_id, revision=revision, token=token
        )
        self._model.eval()
        self._id2label = {
            int(key): str(value).lower()
            for key, value in self._model.config.id2label.items()
        }

    def predict(self, text: str) -> str:
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length",
        )
        with self._torch.no_grad():
            logits = self._model(**encoded).logits[0]
        return self._id2label[int(self._torch.argmax(logits).item())]
