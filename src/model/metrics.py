from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(
    eval_pred, id2label: dict[int, str] | None = None
) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    label_map = id2label or {0: "phishing", 1: "spam", 2: "legitimate"}
    labels_list = list(range(len(label_map)))

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )
    precision_per_class, recall_per_class, f1_per_class, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=labels_list,
            average=None,
            zero_division=0,
        )
    )

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_weighted": float(f1_weighted),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
    }

    for index, name in label_map.items():
        metrics[f"f1_{name}"] = float(f1_per_class[index])
        metrics[f"recall_{name}"] = float(recall_per_class[index])
        metrics[f"precision_{name}"] = float(precision_per_class[index])

    metrics["phishing_recall"] = float(recall_per_class[0])
    legitimate_mask = labels == 2
    if legitimate_mask.sum() > 0:
        fp_count = (predictions[legitimate_mask] == 0).sum()
        metrics["phishing_fp_rate"] = float(fp_count / legitimate_mask.sum())
    else:
        metrics["phishing_fp_rate"] = 0.0

    return metrics
