from __future__ import annotations

import os
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


def promote_if_threshold(
    model_name: str,
    test_metrics: dict[str, float],
    recall_threshold: float = 0.97,
    f1_threshold: float = 0.90,
) -> bool:
    """Assign @production or @staging alias in MLflow registry. Returns True if promoted."""
    has_databricks = bool(
        os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN")
    )
    if not has_databricks:
        print(
            "[promote] Databricks credentials not found — skipping alias promotion."
        )
        return False

    from mlflow import MlflowClient

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = max(int(v.version) for v in versions)

    phishing_recall = test_metrics.get("test_phishing_recall", 0.0)
    f1_weighted = test_metrics.get("test_f1_weighted", 0.0)
    promoted = phishing_recall >= recall_threshold and f1_weighted >= f1_threshold

    if promoted:
        try:
            client.delete_registered_model_alias(model_name, "production")
        except Exception:
            pass
        client.set_registered_model_alias(model_name, "production", str(latest_version))
    else:
        client.set_registered_model_alias(model_name, "staging", str(latest_version))

    return promoted


def push_to_hub(
    save_path: Path,
    hf_repo_id: str,
    hf_token: str,
    test_metrics: dict[str, float],
    mlflow_version: str | None = None,
    artifact_dir: Path | None = None,
) -> str:
    """Push model + tokenizer to HuggingFace Hub.

    After pushing weights and artifacts, moves the ``production`` git tag to
    the new commit so the app can always load ``revision="production"`` without
    ever needing to track a commit SHA.

    Returns the HF repo URL (``https://huggingface.co/{hf_repo_id}``).
    """
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

    # Move the `production` tag to the current HEAD of main.
    # The app loads the model with revision="production" and never needs
    # to track commit SHAs — this tag only advances on an explicit promotion.
    api.create_tag(
        repo_id=hf_repo_id,
        tag="production",
        revision="main",
        token=hf_token,
        exist_ok=True,  # overwrites the tag if it already exists
    )
    print(f"[registry] HF tag 'production' → {hf_repo_id}@main")

    return f"https://huggingface.co/{hf_repo_id}"
