from __future__ import annotations

from pathlib import Path

from transformers import TrainingArguments

from src.config.training_config import TrainingConfig


def compute_warmup_steps(train_size: int, batch_size: int, num_epochs: int) -> int:
    total_steps = (train_size // batch_size) * num_epochs
    return int(0.1 * total_steps)


def create_training_args(
    config: TrainingConfig,
    run_name: str,
    output_dir: str | Path,
    train_size: int,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    warmup_steps: int | None = None,
) -> TrainingArguments:
    resolved_epochs = num_epochs or config.num_epochs
    resolved_batch_size = batch_size or config.batch_size
    resolved_learning_rate = learning_rate or config.learning_rate
    resolved_weight_decay = weight_decay or config.weight_decay
    resolved_warmup = warmup_steps
    if resolved_warmup is None:
        resolved_warmup = compute_warmup_steps(
            train_size, resolved_batch_size, resolved_epochs
        )

    return TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=resolved_epochs,
        per_device_train_batch_size=resolved_batch_size,
        per_device_eval_batch_size=resolved_batch_size * 2,
        learning_rate=resolved_learning_rate,
        weight_decay=resolved_weight_decay,
        warmup_steps=resolved_warmup,
        fp16=config.use_fp16,
        bf16=config.use_bf16,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        save_total_limit=2,
        report_to="mlflow",
        seed=config.seed,
        data_seed=config.seed,
    )
