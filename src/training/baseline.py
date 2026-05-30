from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import EarlyStoppingCallback, TrainingArguments

from src.config.training_config import RuntimeState, TrainingConfig
from src.data.tokenizer import TokenizedSplits
from src.model.builder import compute_class_weights, load_model
from src.model.metrics import compute_metrics
from src.model.trainer import WeightedTrainer
from src.training.args import compute_warmup_steps, create_training_args


@dataclass(slots=True)
class BaselineSetup:
    run_name: str
    output_dir: Path
    training_args: TrainingArguments
    class_weights: torch.Tensor
    trainer: WeightedTrainer
    run_config: dict[str, Any]


def build_run_config(
    config: TrainingConfig,
    runtime: RuntimeState,
    train_size: int,
    val_size: int,
    test_size: int,
    warmup_steps: int,
) -> dict[str, Any]:
    return {
        "base_model": config.model_name,
        "max_length": config.max_length,
        "num_labels": config.num_labels,
        "batch_size": config.batch_size,
        "num_epochs": config.num_epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_steps": warmup_steps,
        "class_weight_strategy": config.class_weight_strategy,
        "phishing_boost": config.phishing_boost,
        "gamma": config.gamma,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "phase": "baseline",
        "precision": "fp16" if config.use_fp16 else "fp32",
        "device": runtime.device,
        "runtime": runtime.runtime_env,
    }


def prepare_baseline_training(
    train_df: pd.DataFrame,
    tokenized_splits: TokenizedSplits,
    config: TrainingConfig,
    runtime: RuntimeState,
) -> BaselineSetup:
    run_name = f"baseline-v{runtime.run_date}"
    output_dir = runtime.output_dir / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = compute_warmup_steps(
        len(tokenized_splits.train),
        config.batch_size,
        config.num_epochs,
    )
    training_args = create_training_args(
        config=config,
        run_name=run_name,
        output_dir=output_dir,
        train_size=len(tokenized_splits.train),
        warmup_steps=warmup_steps,
    )
    class_weights = compute_class_weights(
        train_df["label"].to_numpy(),
        num_labels=config.num_labels,
        strategy=config.class_weight_strategy,
    )
    model = load_model(config, hf_token=runtime.hf_token)
    trainer = WeightedTrainer(
        class_weights=class_weights,
        phishing_boost=config.phishing_boost,
        gamma=config.gamma,
        model=model,
        args=training_args,
        train_dataset=tokenized_splits.train,
        eval_dataset=tokenized_splits.val,
        compute_metrics=lambda eval_pred: compute_metrics(
            eval_pred, id2label=config.id2label
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    run_config = build_run_config(
        config=config,
        runtime=runtime,
        train_size=len(tokenized_splits.train),
        val_size=len(tokenized_splits.val),
        test_size=len(tokenized_splits.test),
        warmup_steps=warmup_steps,
    )
    return BaselineSetup(
        run_name=run_name,
        output_dir=output_dir,
        training_args=training_args,
        class_weights=class_weights,
        trainer=trainer,
        run_config=run_config,
    )
