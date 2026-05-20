"""MLflow experiment tracking, model registry, and HuggingFace Hub publication."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.tracking

from src.config.training_config import RuntimeState, TrainingConfig


@dataclass(slots=True)
class ModelInfo:
    registered_model_version: str | int


def setup_mlflow(state: RuntimeState) -> str:
    """Configure MLflow tracking URI and create/set experiment.

    Uses Databricks Unity Catalog when credentials are present; falls back
    to a local file-based tracking store under output_dir/mlruns.

    Returns the experiment path string.
    """
    if state.databricks_host and state.databricks_token:
        os.environ.setdefault("DATABRICKS_HOST", state.databricks_host)
        os.environ.setdefault("DATABRICKS_TOKEN", state.databricks_token)
        mlflow.set_tracking_uri("databricks")
        experiment_path = (
            f"/Users/{state.databricks_email}/{state.mlflow_experiment_name}"
        )
        mlflow.set_experiment(experiment_path)
    else:
        tracking_dir = state.output_dir / "mlruns"
        tracking_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{tracking_dir.resolve()}")
        experiment_path = state.mlflow_experiment_name
        mlflow.set_experiment(experiment_path)

    return experiment_path


def register_model(
    run_id: str,
    save_path: Path,
    model_name: str,
    config: TrainingConfig,
) -> ModelInfo:
    """Log saved model artifacts into the MLflow run and register in the registry.

    Logs the contents of save_path as a 'model' artifact on the completed run,
    then creates a registered model version pointing to it.
    """
    client = mlflow.tracking.MlflowClient()

    # Attach the saved model directory to the completed run as an artifact.
    client.log_artifacts(run_id, str(save_path), artifact_path="model")

    # Ensure the registered model exists (idempotent).
    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        pass  # already exists

    mv = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
    )
    return ModelInfo(registered_model_version=mv.version)


def promote_if_threshold(
    model_name: str,
    test_metrics: dict[str, float],
    recall_threshold: float = 0.97,
    f1_threshold: float = 0.90,
) -> bool:
    """Assign @production alias if metrics meet both thresholds, else @staging.

    Returns True if the model was promoted to @production.
    """
    f1 = test_metrics.get("test_f1_weighted", 0.0)
    recall = test_metrics.get("test_phishing_recall", 0.0)
    promoted = f1 >= f1_threshold and recall >= recall_threshold

    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_registered_model(model_name).latest_versions
        if not versions:
            return False
        version = versions[-1].version
        alias = "production" if promoted else "staging"
        client.set_registered_model_alias(model_name, alias, version)
    except mlflow.exceptions.MlflowException:
        pass

    return promoted


def push_to_hub(
    save_path: Path,
    hf_repo_id: str,
    hf_token: str,
    test_metrics: dict[str, float],
    mlflow_version: str,
    artifact_dir: Path,
) -> str:
    """Push model, tokenizer, and evaluation artifacts to HuggingFace Hub.

    Uploads the full save_path directory plus any evaluation artefacts
    (confusion matrix image, classification report) from artifact_dir.
    Returns the repo URL.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hf_repo_id, exist_ok=True, private=False)

    commit_message = (
        f"MLflow v{mlflow_version} — "
        f"F1={test_metrics.get('test_f1_weighted', 0):.4f} "
        f"phishing_recall={test_metrics.get('test_phishing_recall', 0):.4f}"
    )

    # Push model + tokenizer files.
    api.upload_folder(
        repo_id=hf_repo_id,
        folder_path=str(save_path),
        commit_message=commit_message,
    )

    # Push evaluation artefacts if they exist.
    for filename in ("confusion_matrix.png", "classification_report.txt"):
        artefact = Path(artifact_dir) / filename
        if artefact.exists():
            api.upload_file(
                repo_id=hf_repo_id,
                path_or_fileobj=str(artefact),
                path_in_repo=f"eval/{filename}",
            )

    return f"https://huggingface.co/{hf_repo_id}"
