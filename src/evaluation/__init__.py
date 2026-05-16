from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.config.training_config import ID2LABEL

if TYPE_CHECKING:
    from transformers import Trainer


def evaluate_on_test(
    trainer: Trainer,
    test_dataset: Any,
    metric_key_prefix: str = "test",
) -> dict[str, float]:
    """Evaluate trainer on a held-out split and return the metrics dict.

    Drops EarlyStoppingCallback before evaluation so it does not interfere
    with the final scoring pass.
    """
    from transformers import EarlyStoppingCallback

    trainer.callback_handler.callbacks = [
        cb
        for cb in trainer.callback_handler.callbacks
        if not isinstance(cb, EarlyStoppingCallback)
    ]
    metrics: dict[str, float] = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix=metric_key_prefix,
    )
    return metrics


def build_error_dataframe(
    test_df: pd.DataFrame,
    raw_predictions: np.ndarray,
    id2label: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Annotate test_df with predictions, per-class probabilities, and a correctness flag."""
    from scipy.special import softmax as sp_softmax

    label_map = id2label or ID2LABEL
    probs = sp_softmax(raw_predictions, axis=-1)
    pred_labels = np.argmax(raw_predictions, axis=-1)

    result = test_df.copy()
    result["predicted"] = pred_labels
    result["pred_label_name"] = result["predicted"].map(label_map)
    result["true_label_name"] = result["label"].map(label_map)
    result["correct"] = result["label"] == result["predicted"]
    result["confidence"] = probs.max(axis=1)
    result["pred_phishing_prob"] = probs[:, 0]
    result["pred_spam_prob"] = probs[:, 1]
    result["pred_legit_prob"] = probs[:, 2]
    return result


def confusion_matrix_arrays(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    num_labels: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw counts, row-normalised) confusion matrix pair."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_labels)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    return cm, cm_norm


def save_classification_report(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    output_dir: Path,
    id2label: dict[int, str] | None = None,
) -> Path:
    """Write a text classification report to output_dir/classification_report.txt."""
    from sklearn.metrics import classification_report

    label_map = id2label or ID2LABEL
    target_names = [label_map[i] for i in sorted(label_map)]
    report = classification_report(true_labels, pred_labels, target_names=target_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    return report_path
