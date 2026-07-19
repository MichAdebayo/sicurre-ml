from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.training_config import RuntimeState, TrainingConfig


def setup_mlflow(runtime_state: RuntimeState) -> str:
    """Configure MLflow tracking URI and experiment. Returns the experiment path."""
    import mlflow

    host = runtime_state.databricks_host
    token = runtime_state.databricks_token

    if host and token:
        os.environ["DATABRICKS_HOST"] = host
        os.environ["DATABRICKS_TOKEN"] = token
        experiment_path = (
            f"/Users/{runtime_state.databricks_email}/{runtime_state.mlflow_experiment_name}"
            if runtime_state.databricks_email
            else runtime_state.mlflow_experiment_name
        )
        mlflow.set_tracking_uri("databricks")
        print(f"[mlflow] Tracking → Databricks  experiment: {experiment_path}")
    else:
        experiment_path = runtime_state.mlflow_experiment_name
        mlflow.set_tracking_uri("file:./mlruns")
        print(
            "[mlflow] Tracking → local file:./mlruns  "
            "(DATABRICKS_HOST/TOKEN not found in secrets — check Kaggle secret names)"
        )

    mlflow.set_experiment(experiment_path)
    return experiment_path


def register_model(
    run_id: str,
    save_path: Path,
    model_name: str,
    config: TrainingConfig,
) -> Any:
    """Log and register a saved transformers model in the MLflow Unity Catalog registry."""
    import mlflow
    import mlflow.transformers
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Load objects explicitly so MLflow receives the model/tokenizer instances
    # rather than a local path string.  When a path string is passed, MLflow
    # tries to resolve it as a HuggingFace repo ID to fetch the model card,
    # which raises a warning because a filesystem path is not a valid repo ID.
    model_obj = AutoModelForSequenceClassification.from_pretrained(str(save_path))
    tokenizer_obj = AutoTokenizer.from_pretrained(str(save_path))

    # setup_mlflow() sets DATABRICKS_HOST/TOKEN when credentials are available.
    # Only point at Unity Catalog when those env vars are actually present;
    # otherwise log the model locally without registering (no hard crash on
    # runs where Databricks secrets have not been configured in Kaggle).
    has_databricks = bool(
        os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN")
    )
    if has_databricks:
        mlflow.set_registry_uri("databricks-uc")
        registered_name: str | None = model_name
    else:
        print(
            "[registry] Databricks credentials not found — "
            "logging model artifact without Unity Catalog registration."
        )
        registered_name = None

    with mlflow.start_run(run_id=run_id):
        model_info = mlflow.transformers.log_model(
            transformers_model={"model": model_obj, "tokenizer": tokenizer_obj},
            name="model",
            task="text-classification",
            registered_model_name=registered_name,
            pip_requirements=[
                f"transformers=={transformers.__version__}",
                "torch>=2.2",
                "sentencepiece",
            ],
        )
    return model_info


def stage_candidate(model_name: str, model_version: str) -> None:
    """Assign the MLflow candidate alias without changing production."""
    has_databricks = bool(
        os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN")
    )
    if not has_databricks:
        print("[registry] Databricks credentials not found — skipping candidate alias.")
        return

    import mlflow
    from mlflow import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "candidate", model_version)
    print(f"[registry] MLflow candidate → {model_name} v{model_version}")


def promote_registered_model(
    model_name: str,
    model_version: str,
    *,
    approved: bool,
) -> None:
    """Move MLflow production only after an external reviewed gate passes."""
    if not approved:
        raise ValueError("Explicit approval is required for production promotion")
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "production", model_version)
    print(f"[registry] MLflow production → {model_name} v{model_version}")


def export_to_onnx(
    save_path: Path,
    opset: int = 17,
) -> Path:
    """Export the saved PyTorch model to ONNX before candidate publication.

    Uses torch directly — no optimum dependency — which avoids the
    transformers/optimum version incompatibility on Kaggle (optimum 1.x
    imports ``is_tf_available`` which was removed in transformers 5.x).

    Returns the local path to the exported model.onnx (stored inside save_path).
    """
    import torch
    from transformers import AutoModelForSequenceClassification

    stable_path = save_path / "model.onnx"

    print(f"[onnx-export] Loading model from {save_path} ...")
    model = AutoModelForSequenceClassification.from_pretrained(str(save_path))
    model.eval()

    # Wrapper so torch.onnx.export receives a plain tensor output, not
    # the SequenceClassifierOutput dataclass that PyTorch can't trace cleanly.
    class _LogitsWrapper(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapper = _LogitsWrapper(model)
    max_length = 256
    dummy_ids  = torch.ones(1, max_length, dtype=torch.long)
    dummy_mask = torch.ones(1, max_length, dtype=torch.long)

    print(f"[onnx-export] Exporting to ONNX (opset={opset}) ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_mask),
            str(stable_path),
            opset_version=opset,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits":         {0: "batch_size"},
            },
        )
    print(f"[onnx-export] Written to {stable_path}")

    return stable_path


@dataclass(frozen=True, slots=True)
class HfPublication:
    repo_url: str
    revision: str


def publish_candidate_to_hub(
    save_path: Path,
    hf_repo_id: str,
    hf_token: str,
    test_metrics: dict[str, float],
    mlflow_version: str | None = None,
    artifact_dir: Path | None = None,
) -> HfPublication:
    """Publish a complete candidate and return its immutable Hub revision."""
    from huggingface_hub import HfApi
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    commit_msg = (
        f"MLflow v{mlflow_version or '?'} | "
        f"F1={test_metrics.get('test_f1_weighted', 0.0):.4f} | "
        f"PhishingRecall={test_metrics.get('test_phishing_recall', 0.0):.4f}"
    )

    model = AutoModelForSequenceClassification.from_pretrained(str(save_path))
    tokenizer = AutoTokenizer.from_pretrained(str(save_path))
    model.push_to_hub(hf_repo_id, token=hf_token, commit_message=commit_msg)
    tokenizer.push_to_hub(hf_repo_id, token=hf_token, commit_message=commit_msg)

    api = HfApi()

    if artifact_dir is not None:
        for artifact_name in ("confusion_matrix.png", "classification_report.txt"):
            artifact_path = artifact_dir / artifact_name
            if artifact_path.exists():
                api.upload_file(
                    path_or_fileobj=str(artifact_path),
                    path_in_repo=artifact_name,
                    repo_id=hf_repo_id,
                    token=hf_token,
                    commit_message=f"Upload eval artifact: {artifact_name}",
                )

    onnx_path = save_path / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError("model.onnx must be exported before candidate publication")
    api.upload_file(
        path_or_fileobj=str(onnx_path),
        path_in_repo="model.onnx",
        repo_id=hf_repo_id,
        token=hf_token,
        commit_message="Upload candidate ONNX artifact",
    )

    revision = api.model_info(repo_id=hf_repo_id, revision="main", token=hf_token).sha
    if not revision:
        raise RuntimeError("Hugging Face did not return an immutable candidate revision")
    print(f"[registry] HF candidate published → {hf_repo_id}@{revision}")
    return HfPublication(
        repo_url=f"https://huggingface.co/{hf_repo_id}",
        revision=revision,
    )


def promote_hf_revision(
    hf_repo_id: str,
    hf_token: str,
    revision: str,
    *,
    approved: bool,
) -> str:
    """Move and verify the HF production tag at an exact immutable revision."""
    if not approved:
        raise ValueError("Explicit approval is required for production promotion")
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        api.delete_tag(repo_id=hf_repo_id, tag="production", token=hf_token)
    except Exception as exc:
        if "404" not in str(exc) and "not found" not in str(exc).lower():
            raise
    api.create_tag(
        repo_id=hf_repo_id,
        tag="production",
        revision=revision,
        token=hf_token,
    )
    resolved = api.model_info(
        repo_id=hf_repo_id,
        revision="production",
        token=hf_token,
    ).sha
    if resolved != revision:
        raise RuntimeError(
            f"HF production verification failed: expected {revision}, resolved {resolved}"
        )
    print(f"[registry] HF production → {hf_repo_id}@{revision}")
    return resolved
