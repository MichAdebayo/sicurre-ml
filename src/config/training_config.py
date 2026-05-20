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
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        missing: list[str] = []
        for key in _SECRET_KEYS:
            try:
                secrets[key] = client.get_secret(key)
            except Exception:
                secrets[key] = None
                missing.append(key)
        if missing:
            print(
                f"[secrets] WARNING: {len(missing)} secret(s) not attached to this "
                f"notebook: {missing}\n"
                "         → In the Kaggle notebook UI go to Environment → Secrets "
                "and toggle each one on."
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


def create_training_config(device: str) -> TrainingConfig:
    return TrainingConfig(
        batch_size=16 if device == "cuda" else 8, use_fp16=device == "cuda"
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
