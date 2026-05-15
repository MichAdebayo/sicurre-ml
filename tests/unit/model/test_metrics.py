import numpy as np

from src.model.metrics import compute_metrics


def test_compute_metrics_returns_expected_keys() -> None:
    logits = np.array([[4.0, 1.0, 0.2], [0.1, 3.0, 0.2], [0.1, 0.2, 2.5]])
    labels = np.array([0, 1, 2])
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 1.0
    assert "phishing_recall" in metrics
    assert "phishing_fp_rate" in metrics
