"""Stage 3 — ONNX CamemBERTa classifier.

Loads model.onnx + tokenizer from the HuggingFace Hub on first call and
caches them in memory. Re-checks the Hub commit SHA on startup so the
container automatically picks up a newly promoted model after a restart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class OnnxResult:
    label: str          # "phishing" or "safe"
    confidence: float   # 0.0–1.0
    raw_scores: dict[str, float]


# ---------------------------------------------------------------------------
# Model cache helpers
# ---------------------------------------------------------------------------

def _model_cache_dir() -> Path:
    base = Path(os.getenv("ONNX_MODEL_CACHE_DIR", "/tmp/sicurre_onnx"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _hf_repo_id() -> str:
    username = os.environ["HF_USERNAME"]
    repo = os.environ["REPO_NAME"]
    return f"{username}/{repo}"


def _pull_from_hub() -> Path:
    """Download model.onnx + tokenizer from HF Hub into local cache.

    Checks the Hub's current commit SHA against a stored sha.txt; skips
    download if already up-to-date (startup hot path after first load).
    """
    from huggingface_hub import HfApi, snapshot_download

    repo_id = _hf_repo_id()
    cache_dir = _model_cache_dir()
    sha_file = cache_dir / "sha.txt"
    hf_token = os.getenv("HF_TOKEN")

    api = HfApi()
    info = api.repo_info(repo_id, repo_type="model", token=hf_token)
    remote_sha: str = info.sha or ""

    if sha_file.exists() and (cache_dir / "model.onnx").exists():
        local_sha = sha_file.read_text().strip()
        if local_sha == remote_sha:
            print(f"[onnx] Cache up-to-date (sha={remote_sha[:8]}).")
            return cache_dir

    print(f"[onnx] Downloading model from {repo_id} (sha={remote_sha[:8]})…")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(cache_dir),
        token=hf_token,
        ignore_patterns=["*.bin", "*.safetensors", "flax_model.*", "tf_model.*"],
    )
    sha_file.write_text(remote_sha)
    print(f"[onnx] Download complete → {cache_dir}")
    return cache_dir


@lru_cache(maxsize=1)
def _load_session_and_tokenizer() -> tuple[object, object, dict[int, str]]:
    """Load (ort.InferenceSession, tokenizer, id2label). Cached after first call."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_dir = _pull_from_hub()

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = int(os.getenv("ONNX_NUM_THREADS", "2"))
    sess_opts.inter_op_num_threads = int(os.getenv("ONNX_NUM_THREADS", "2"))

    session = ort.InferenceSession(
        str(model_dir / "model.onnx"),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # Read id2label from config.json if present
    import json
    config_path = model_dir / "config.json"
    id2label: dict[int, str] = {0: "safe", 1: "phishing"}
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        raw = cfg.get("id2label", {})
        id2label = {int(k): v for k, v in raw.items()}

    print(f"[onnx] Session ready. Labels: {id2label}")
    return session, tokenizer, id2label


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify_onnx(text: str, max_length: int = 256) -> OnnxResult:
    """Run the ONNX CamemBERTa model on *text*."""
    import numpy as np
    import onnxruntime as ort

    raw_session, tokenizer, id2label = _load_session_and_tokenizer()
    session: ort.InferenceSession = raw_session  # type: ignore[assignment]

    inputs = tokenizer(  # type: ignore[operator]
        text,
        return_tensors="np",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in [inp.name for inp in session.get_inputs()]:
        ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

    logits = session.run(None, ort_inputs)[0][0]  # shape: (num_labels,)

    # Softmax
    exp_l = np.exp(logits - logits.max())
    probs = exp_l / exp_l.sum()

    best_idx = int(probs.argmax())
    raw_scores = {id2label[i]: float(probs[i]) for i in range(len(probs))}

    return OnnxResult(
        label=id2label[best_idx],
        confidence=float(probs[best_idx]),
        raw_scores=raw_scores,
    )


def get_model_version() -> str:
    cache_dir = _model_cache_dir()
    sha_file = cache_dir / "sha.txt"
    if sha_file.exists():
        return sha_file.read_text().strip() or "unknown"
    return os.getenv("MODEL_SHA", os.getenv("ONNX_MODEL_SHA", "unknown"))
