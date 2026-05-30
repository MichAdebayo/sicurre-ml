import tempfile

import torch
import pytest

from src.model.trainer import WeightedTrainer


class _TinyModel(torch.nn.Module):
    """Minimal 3-class classifier used purely for trainer unit tests."""

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch = input_ids.shape[0] if input_ids is not None else 1
        logits = torch.zeros(batch, 3)
        return type("Out", (), {"logits": logits, "loss": None})()


def _make_trainer(tmp_path) -> tuple[WeightedTrainer, _TinyModel]:
    from transformers import TrainingArguments

    weights = torch.tensor([1.0, 1.0, 1.0])
    model = _TinyModel()
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none")
    trainer = WeightedTrainer(class_weights=weights, model=model, args=args)
    return trainer, model


def test_weighted_trainer_init_stores_attributes(tmp_path) -> None:
    trainer, _ = _make_trainer(tmp_path)
    assert trainer.phishing_boost == 2.0
    assert trainer.gamma == 1.5
    assert trainer.log_every_n == 50
    assert trainer.global_step_logs == []
    assert trainer.class_weights is not None


def test_weighted_trainer_tokenizer_kwarg_is_renamed(tmp_path) -> None:
    from transformers import TrainingArguments

    model = _TinyModel()
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none")
    # Passing 'tokenizer' should be silently renamed to 'processing_class'.
    trainer = WeightedTrainer(
        model=model, args=args, class_weights=None, tokenizer=None
    )
    assert trainer is not None


def test_weighted_trainer_compute_loss_returns_scalar(tmp_path) -> None:
    trainer, model = _make_trainer(tmp_path)
    inputs = {
        "input_ids": torch.zeros(4, 5, dtype=torch.long),
        "attention_mask": torch.ones(4, 5, dtype=torch.long),
        "labels": torch.tensor([0, 1, 2, 0]),
    }
    # global_step starts at 0 → 0 % 50 == 0 → logging branch executes.
    loss = trainer.compute_loss(model, inputs)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert loss.item() >= 0


def test_weighted_trainer_compute_loss_return_outputs(tmp_path) -> None:
    trainer, model = _make_trainer(tmp_path)
    inputs = {
        "input_ids": torch.zeros(2, 5, dtype=torch.long),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
        "labels": torch.tensor([0, 1]),
    }
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    assert isinstance(loss, torch.Tensor)
    assert hasattr(outputs, "logits")


def test_weighted_trainer_compute_loss_without_class_weights(tmp_path) -> None:
    from transformers import TrainingArguments

    model = _TinyModel()
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none")
    trainer = WeightedTrainer(class_weights=None, model=model, args=args)
    inputs = {
        "input_ids": torch.zeros(2, 5, dtype=torch.long),
        "labels": torch.tensor([0, 1]),
    }
    loss = trainer.compute_loss(model, inputs)
    assert loss.item() >= 0
