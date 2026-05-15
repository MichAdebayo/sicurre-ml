from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

from src.config.training_config import TrainingConfig


def compute_class_weights(
    labels: np.ndarray,
    num_labels: int,
    strategy: str = "inverse_freq",
) -> torch.Tensor:
    match strategy:
        case "none":
            weights = np.ones(num_labels)
        case "inverse_freq":
            counts = Counter(labels)
            total = len(labels)
            weights = np.array(
                [total / (num_labels * counts[index]) for index in range(num_labels)]
            )
        case "phishing_heavy":
            counts = Counter(labels)
            total = len(labels)
            weights = np.array(
                [total / (num_labels * counts[index]) for index in range(num_labels)]
            )
            weights[0] *= 1.5
        case _:
            raise ValueError(f"Unknown strategy: {strategy}")

    weights = weights / weights.sum() * num_labels
    return torch.tensor(weights, dtype=torch.float32)


def load_model(config: TrainingConfig, hf_token: str | None = None):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            token=hf_token,
            num_labels=config.num_labels,
            id2label=config.id2label,
            label2id=config.label2id,
        )
