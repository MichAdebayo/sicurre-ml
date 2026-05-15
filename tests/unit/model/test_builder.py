import numpy as np
import pytest

from src.model.builder import compute_class_weights


def test_compute_class_weights_normalizes_values() -> None:
    weights = compute_class_weights(
        np.array([0, 0, 1, 2]), num_labels=3, strategy="inverse_freq"
    )
    assert pytest.approx(float(weights.sum()), rel=1e-6) == 3.0


def test_compute_class_weights_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError):
        compute_class_weights(np.array([0, 1, 2]), num_labels=3, strategy="bad")
