from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
from dotenv import load_dotenv

RuntimeEnv = Literal["local", "colab", "kaggle"]

LABEL_NAMES = ["phishing", "spam", "legitimate"]
ID2LABEL = {0: "phishing", 1: "spam", 2: "legitimate"}
LABEL2ID = {"phishing": 0, "spam": 1, "legitimate": 2}

_SECRET_KEYS = (
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_EMAIL",
    "MLFLOW_EXPERIMENT_NAME",
    "HF_TOKEN",
    "HF_USERNAME",
    "REPO_NAME",
    "R2_EVALUATION_ACCESS_KEY_ID",
    "R2_EVALUATION_SECRET_ACCESS_KEY",
    "R2_EVALUATION_ENDPOINT",
    "R2_EVALUATION_BUCKET_NAME",
    "SICURRE_INTERNAL_API_KEY",
    "SICURRE_CALLBACK_BASE_URL",
)


@dataclass(slots=True)
class TrainingConfig:
    model_name: str = "almanach/camembertav2-base"
    num_labels: int = 3
    label_names: list[str] = field(default_factory=lambda: LABEL_NAMES.copy())
    id2label: dict[int, str] = field(default_factory=lambda: ID2LABEL.copy())
    label2id: dict[str, int] = field(default_factory=lambda: LABEL2ID.copy())
    max_length: int = 256
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    # Four, matching the epoch budget the incumbent (v15) was trained with, so
    # candidates are comparable to it rather than systematically undertrained.
    #
    # This was briefly lowered to three while training on a ~3x augmented corpus,
    # where three passes already exceed the samples a four-epoch run sees on the
    # base data. That reasoning does not carry over to base-sized datasets: a
    # three-epoch run on 26k rows sees 78k samples against the incumbent's 104k,
    # a 25% deficit that shows up as a weighted-F1 gap and gets misread as a
    # worse recipe. Scale epochs to the corpus; do not carry a number across.
    num_epochs: int = 4
    seed: int = 42
    use_fp16: bool = False
    use_bf16: bool = False
    enable_quantization: bool = False
    quantization_mode: str | None = None
    class_weight_strategy: str = "inverse_freq"
    phishing_boost: float = 2.0
    gamma: float = 1.5
    mlflow_model_name: str = "main.sicurre.phishing-detector"


@dataclass(slots=True)
class RuntimeState:
    runtime_env: RuntimeEnv
    device: str
    use_tpu: bool
    run_date: str
    data_dir: Path
    output_dir: Path
    secrets: dict[str, str | None]
    hf_token: str | None
    databricks_host: str | None
    databricks_token: str | None
    databricks_email: str | None
    mlflow_experiment_name: str


def detect_runtime() -> RuntimeEnv:
    if os.path.exists("/kaggle/working"):
        return "kaggle"
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return "local"
    return "colab"


def detect_device() -> tuple[str, bool]:
    try:
        import torch_xla.core.xla_model as xm

        return str(xm.xla_device()), True
    except Exception:
        if torch.cuda.is_available():
            return "cuda", False
        if torch.backends.mps.is_available():
            return "mps", False
        return "cpu", False


def _empty_secrets() -> dict[str, str | None]:
    return {key: None for key in _SECRET_KEYS}


def load_secrets(runtime_env: RuntimeEnv) -> dict[str, str | None]:
    secrets = _empty_secrets()

    if runtime_env == "kaggle":
        # CI push: GitHub secrets are injected as os.environ in a prepended cell
        # before this code runs.  Interactive run: fall back to UserSecretsClient.
        _usk: object = None
        _usk_init = False
        missing: list[str] = []
        for key in _SECRET_KEYS:
            if os.environ.get(key):
                secrets[key] = os.environ[key]
                continue
            if not _usk_init:
                try:
                    from kaggle_secrets import UserSecretsClient

                    _usk = UserSecretsClient()
                except Exception:
                    pass
                _usk_init = True
            if _usk is not None:
                try:
                    secrets[key] = _usk.get_secret(key)  # type: ignore[attr-defined]
                except Exception:
                    secrets[key] = None
                    missing.append(key)
            else:
                secrets[key] = None
                missing.append(key)
        if missing:
            print(
                f"[secrets] WARNING: {len(missing)} secret(s) not available: {missing}\n"
                "         → CI push: verify all 7 secrets are set in GitHub repo secrets.\n"
                "         → Interactive: attach in Kaggle UI → Environment → Secrets."
            )
        return secrets

    if runtime_env == "colab":
        from google.colab import userdata

        missing = []
        for key in _SECRET_KEYS:
            try:
                secrets[key] = userdata.get(key)
            except Exception:
                secrets[key] = None
                missing.append(key)
        if missing:
            print(
                f"[secrets] WARNING: {len(missing)} secret(s) not found "
                f"in Colab userdata: {missing}"
            )
        return secrets

    load_dotenv()
    for key in _SECRET_KEYS:
        secrets[key] = os.getenv(key)
    return secrets


def enable_deterministic_training(seed: int = 42) -> dict[str, object]:
    """Pin every source of run-to-run randomness we can reach.

    Setting `seed` alone is not enough on a GPU. cuDNN benchmarks several
    convolution algorithms on first use and keeps whichever was fastest on that
    machine at that moment, and several CUDA kernels accumulate with atomics
    whose order is not fixed. Two runs of identical code on identical data
    therefore produce different weights.

    That is not theoretical here. Two runs of this pipeline with the same
    dataset, the same four epochs and the same 168 recorded parameters produced
    phishing recall of 0.68 and 0.92, and legitimate false-positive rates of
    0.28 and 0.56. The promotion gate permits no regression on either number, so
    without this it is comparing draws rather than models.

    `warn_only=True` is deliberate: a handful of operations have no deterministic
    implementation, and raising on them would abort training instead of merely
    leaving those few ops unpinned. This removes the large, dominant sources.

    CUBLAS_WORKSPACE_CONFIG must be set before the first CUDA context is created,
    so call this early - before the model or any tensor is built.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001 - older torch without the kwarg
        torch.use_deterministic_algorithms(True)

    return {
        "seed": seed,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": True,
    }


def create_training_config(device: str) -> TrainingConfig:
    return TrainingConfig(
        batch_size=16 if device == "cuda" else 8,
        use_fp16=device == "cuda",
    )


def _resolve_data_dir(runtime_env: RuntimeEnv) -> Path:
    if runtime_env == "colab":
        return Path("/content/drive/MyDrive/sicurre/data/final")
    if runtime_env == "kaggle":
        return Path(
            "/kaggle/input/datasets/michaeladebayo99/sicurre-finetuning-dataset"
        )
    return Path("data/final")


def _resolve_output_dir(runtime_env: RuntimeEnv, run_date: str) -> Path:
    if runtime_env == "colab":
        return (
            Path("/content/drive/MyDrive/sicurre/models/camembertav2-phishing-fr")
            / f"v{run_date}"
        )
    if runtime_env == "kaggle":
        return Path("/kaggle/working/models/camembertav2-phishing-fr") / f"v{run_date}"
    return Path("models/camembertav2-phishing-fr") / f"v{run_date}"


def build_runtime_state(
    runtime_env: RuntimeEnv | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> RuntimeState:
    resolved_runtime = runtime_env or detect_runtime()
    device, use_tpu = detect_device()
    secrets = load_secrets(resolved_runtime)
    run_date = datetime.now().strftime("%Y%m%d")
    resolved_data_dir = data_dir or _resolve_data_dir(resolved_runtime)
    resolved_output_dir = output_dir or _resolve_output_dir(resolved_runtime, run_date)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    return RuntimeState(
        runtime_env=resolved_runtime,
        device=device,
        use_tpu=use_tpu,
        run_date=run_date,
        data_dir=resolved_data_dir,
        output_dir=resolved_output_dir,
        secrets=secrets,
        hf_token=secrets["HF_TOKEN"],
        databricks_host=secrets["DATABRICKS_HOST"],
        databricks_token=secrets["DATABRICKS_TOKEN"],
        databricks_email=secrets["DATABRICKS_EMAIL"],
        mlflow_experiment_name=(
            secrets["MLFLOW_EXPERIMENT_NAME"] or "sicurre-camembertav2-phishing-fr"
        ),
    )
