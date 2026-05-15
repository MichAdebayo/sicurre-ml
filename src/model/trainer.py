from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import Trainer


class WeightedTrainer(Trainer):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        phishing_boost: float = 2.0,
        gamma: float = 1.5,
        log_every_n: int = 50,
        **kwargs: Any,
    ) -> None:
        if "tokenizer" in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.phishing_boost = phishing_boost
        self.gamma = gamma
        self.log_every_n = log_every_n
        self.global_step_logs: list[dict[str, float | int]] = []

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor | tuple[Tensor, Any]:
        del num_items_in_batch
        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels", None)
        if labels is None:
            labels = model_inputs.pop("label")
        outputs = model(**model_inputs)
        logits = outputs.logits

        weights = self.class_weights
        if weights is not None:
            weights = weights.to(logits.device).clone()
            weights[0] *= self.phishing_boost

        loss_per_sample = F.cross_entropy(
            logits,
            labels,
            weight=weights,
            reduction="none",
            label_smoothing=0.1,
        )

        with torch.no_grad():
            probabilities = F.softmax(logits, dim=-1)
            pt = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_term = (1 - pt) ** self.gamma
        loss = (focal_term * loss_per_sample).mean()

        if self.state.global_step % self.log_every_n == 0:
            predictions = logits.argmax(dim=-1)
            phishing_total = (labels == 0).sum().clamp(min=1)
            non_phishing_total = (labels != 0).sum().clamp(min=1)
            phishing_hits = ((predictions == 0) & (labels == 0)).float().sum()
            phishing_recall = (phishing_hits / phishing_total).item()
            false_positives = ((predictions == 0) & (labels != 0)).float().sum()
            fp_rate = (false_positives / non_phishing_total).item()
            self.global_step_logs.append(
                {
                    "step": int(self.state.global_step),
                    "loss": float(loss.item()),
                    "phishing_recall_batch": float(phishing_recall),
                    "fp_rate_batch": float(fp_rate),
                }
            )

        return (loss, outputs) if return_outputs else loss

    def _load_best_model(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super()._load_best_model()
