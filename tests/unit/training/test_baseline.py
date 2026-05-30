from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.config.training_config import RuntimeState, TrainingConfig
from src.training.baseline import build_run_config


def _make_runtime(tmp_path: Path) -> RuntimeState:
    return RuntimeState(
        runtime_env="local",
        device="cpu",
        use_tpu=False,
        run_date="20250530",
        data_dir=tmp_path,
        output_dir=tmp_path,
        secrets={},
        hf_token=None,
        databricks_host=None,
        databricks_token=None,
        databricks_email=None,
        mlflow_experiment_name="test-experiment",
    )


def test_build_run_config_returns_expected_keys(tmp_path: Path) -> None:
    config = TrainingConfig()
    runtime = _make_runtime(tmp_path)
    result = build_run_config(config, runtime, 100, 20, 10, 50)
    assert result["phase"] == "baseline"
    assert result["train_size"] == 100
    assert result["val_size"] == 20
    assert result["test_size"] == 10
    assert result["warmup_steps"] == 50
    assert result["base_model"] == config.model_name


def test_prepare_baseline_training_builds_setup(tmp_path: Path) -> None:
    from src.data.tokenizer import TokenizedSplits
    from src.training.baseline import prepare_baseline_training

    config = TrainingConfig()
    runtime = _make_runtime(tmp_path)

    mock_dataset = MagicMock()
    mock_dataset.__len__ = MagicMock(return_value=10)
    tokenized_splits = TokenizedSplits(
        tokenizer=MagicMock(),
        train=mock_dataset,
        val=mock_dataset,
        test=mock_dataset,
    )
    train_df = pd.DataFrame(
        {"text": ["a"] * 10, "label": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]}
    )

    with patch("src.training.baseline.load_model", return_value=MagicMock()):
        with patch("src.training.baseline.WeightedTrainer") as MockTrainer:
            MockTrainer.return_value = MagicMock()
            setup = prepare_baseline_training(train_df, tokenized_splits, config, runtime)

    assert "baseline-v" in setup.run_name
    assert "baseline" in str(setup.output_dir)
    assert "phase" in setup.run_config
