"""Evaluation helpers for the Sicurre classifier."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.config.training_config import LABEL_NAMES
from src.model.metrics import compute_metrics


def evaluate_on_test(trainer, test_dataset) -> dict[str, float]:
    """Run prediction on test_dataset and return test_-prefixed metrics.

    Ends any active MLflow run before prediction — the HuggingFace MLflow
    callback may leave the training run open, which prevents clean logging
    of evaluation results into a separate run context.
    """
    import mlflow

    if mlflow.active_run():
        mlflow.end_run()

    predictions = trainer.predict(test_dataset)
    metrics = compute_metrics(
        (predictions.predictions, predictions.label_ids),
        id2label={0: "phishing", 1: "spam", 2: "legitimate"},
    )
    return {f"test_{k}": v for k, v in metrics.items()}


def build_error_dataframe(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Return test_df extended with pred_label, confidence, and correct columns."""
    pred_labels = np.argmax(predictions, axis=-1)
    confidence = predictions[np.arange(len(pred_labels)), pred_labels]
    result = test_df.copy().reset_index(drop=True)
    result["pred_label"] = pred_labels
    result["confidence"] = confidence
    result["correct"] = result["label"] == result["pred_label"]
    return result


def confusion_matrix_arrays(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (counts_cm, row-normalised_cm) for the 3-class classifier."""
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)
    return cm, cm_norm


def save_classification_report(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    output_dir: Path,
) -> Path:
    """Write sklearn classification_report to output_dir/classification_report.txt."""
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=LABEL_NAMES,
        zero_division=0,
    )
    report_path = Path(output_dir) / "classification_report.txt"
    report_path.write_text(report)
    return report_path
