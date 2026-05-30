from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.model.builder import compute_class_weights, load_model
from src.config.training_config import TrainingConfig


def test_compute_class_weights_normalizes_values() -> None:
    weights = compute_class_weights(
        np.array([0, 0, 1, 2]), num_labels=3, strategy="inverse_freq"
    )
    assert pytest.approx(float(weights.sum()), rel=1e-6) == 3.0


def test_compute_class_weights_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError):
        compute_class_weights(np.array([0, 1, 2]), num_labels=3, strategy="bad")


def test_compute_class_weights_none_strategy_returns_uniform() -> None:
    weights = compute_class_weights(np.array([0, 1, 2]), num_labels=3, strategy="none")
    assert weights.shape == (3,)
    # All weights equal before renormalization: 1/1/1 → sum = 3
    assert pytest.approx(float(weights.sum()), rel=1e-6) == 3.0


def test_compute_class_weights_phishing_heavy_boosts_class_zero() -> None:
    labels = np.array([0, 0, 1, 2])
    weights_heavy = compute_class_weights(labels, num_labels=3, strategy="phishing_heavy")
    weights_inv = compute_class_weights(labels, num_labels=3, strategy="inverse_freq")
    # phishing_heavy applies a 1.5× multiplier to class 0 relative to inverse_freq.
    assert float(weights_heavy[0]) > float(weights_inv[0])


def test_load_model_calls_pretrained(monkeypatch) -> None:
    config = TrainingConfig()
    mock_model = MagicMock()
    with patch(
        "src.model.builder.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model,
    ):
        result = load_model(config, hf_token="test-token")
    assert result is mock_model

