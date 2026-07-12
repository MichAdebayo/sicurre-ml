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
    recall_floor: float = 0.97,
    f1_floor: float = 0.90,
    tolerance: float = 0.0,
) -> bool:
    """Assign @production or @staging alias in MLflow registry. Returns True if promoted.

    Promotion logic:
    - If a @production model already exists, the new model must match or beat
      its recall AND F1 minus ``tolerance``.  The floor thresholds still act as
      a minimum safety net — a model that regresses below the floor is never
      promoted even if the bar was already low.
    - ``tolerance`` (default 0.0) allows a small metric drop caused by noise or
      a larger, more varied dataset.  Set via the PROMOTION_TOLERANCE env var.
    - If no @production model exists yet, the floor thresholds alone decide.
    """
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

    new_recall = test_metrics.get("test_phishing_recall", 0.0)
    new_f1 = test_metrics.get("test_f1_weighted", 0.0)

    # --- Compare against current production model if one exists ----------
    prod_recall: float = recall_floor
    prod_f1: float = f1_floor
    has_production = False

    try:
        prod_mv = client.get_model_version_by_alias(model_name, "production")
        if prod_mv.run_id is None:
            raise ValueError("Production model version has no associated MLflow run")
        prod_run = client.get_run(prod_mv.run_id)
        prod_metrics = prod_run.data.metrics
        fetched_recall = prod_metrics.get("test_phishing_recall")
        fetched_f1 = prod_metrics.get("test_f1_weighted")
        if fetched_recall is not None and fetched_f1 is not None:
            # Use production score as the bar, but never drop below the floor
            prod_recall = max(float(fetched_recall), recall_floor)
            prod_f1 = max(float(fetched_f1), f1_floor)
            has_production = True
            print(
                f"[promote] Production baseline — recall={fetched_recall:.4f}  "
                f"f1={fetched_f1:.4f}  (effective bar: recall≥{prod_recall:.4f} f1≥{prod_f1:.4f})"
            )
    except Exception:
        print(
            f"[promote] No @production model found — using floor thresholds "
            f"(recall≥{recall_floor:.2f} f1≥{f1_floor:.2f})"
        )

    promoted = new_recall >= prod_recall - tolerance and new_f1 >= prod_f1 - tolerance
    if has_production and promoted:
        beats_by = f"recall+{new_recall - float(prod_recall):.4f}  f1+{new_f1 - float(prod_f1):.4f}"
        print(f"[promote] New model beats production — {beats_by}")
    elif not promoted:
        tol_note = f" (tolerance={tolerance:.4f})" if tolerance else ""
        print(
            f"[promote] New model did not beat bar{tol_note} — "
            f"recall={new_recall:.4f} (need {prod_recall - tolerance:.4f})  "
            f"f1={new_f1:.4f} (need {prod_f1 - tolerance:.4f})"
        )

    if promoted:
        try:
            client.delete_registered_model_alias(model_name, "production")
        except Exception:
            pass
        client.set_registered_model_alias(model_name, "production", str(latest_version))
    else:
        client.set_registered_model_alias(model_name, "staging", str(latest_version))

    return promoted


def export_to_onnx(
    save_path: Path,
    hf_repo_id: str,
    hf_token: str,
    opset: int = 17,
) -> Path:
    """Export the saved PyTorch model to ONNX via torch.onnx.export and push to HF Hub.

    Uses torch directly — no optimum dependency — which avoids the
    transformers/optimum version incompatibility on Kaggle (optimum 1.x
    imports ``is_tf_available`` which was removed in transformers 5.x).

    Returns the local path to the exported model.onnx (stored inside save_path).
    """
    import torch
    from huggingface_hub import HfApi
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

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(stable_path),
        path_in_repo="model.onnx",
        repo_id=hf_repo_id,
        token=hf_token,
        commit_message=f"Add model.onnx — torch.onnx export (opset {opset})",
    )
    print(f"[onnx-export] Pushed model.onnx → {hf_repo_id}")
    return stable_path


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
