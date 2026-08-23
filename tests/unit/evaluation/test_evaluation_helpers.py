from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.evaluation import (
    build_error_dataframe,
    confusion_matrix_arrays,
    evaluate_on_test,
    save_classification_report,
)


def test_evaluate_on_test_removes_early_stopping_callback() -> None:
    from transformers import EarlyStoppingCallback

    retained = object()

    class FakeTrainer:
        callback_handler = SimpleNamespace(
            callbacks=[EarlyStoppingCallback(), retained]
        )

        def evaluate(self, **kwargs):  # noqa: ANN003, ANN202
            assert kwargs == {"eval_dataset": "held-out", "metric_key_prefix": "final"}
            return {"final_loss": 0.25}

    trainer = FakeTrainer()

    assert evaluate_on_test(trainer, "held-out", "final") == {"final_loss": 0.25}  # type: ignore[arg-type]
    assert trainer.callback_handler.callbacks == [retained]


def test_build_error_dataframe_adds_predictions_and_probabilities() -> None:
    frame = pd.DataFrame({"label": [0, 2], "text": ["a", "b"]})
    logits = np.array([[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]])

    result = build_error_dataframe(frame, logits)

    assert result["predicted"].tolist() == [0, 2]
    assert result["correct"].tolist() == [True, True]
    assert result["pred_label_name"].tolist() == ["phishing", "legitimate"]
    assert all(result["confidence"] > 0.8)


def test_confusion_matrix_returns_raw_and_row_normalized() -> None:
    raw, normalized = confusion_matrix_arrays(
        np.array([0, 0, 1, 2]),
        np.array([0, 1, 1, 2]),
    )

    assert raw.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert normalized[0].tolist() == [0.5, 0.5, 0.0]


def test_classification_report_is_written(tmp_path: Path) -> None:
    report = save_classification_report(
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
        tmp_path / "reports",
    )

    assert report.name == "classification_report.txt"
    assert "phishing" in report.read_text(encoding="utf-8")
