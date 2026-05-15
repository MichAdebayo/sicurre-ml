---
name: training-pipeline
description: "Fine-tuning CamemBERTav2 with WeightedTrainer, focal loss, class weights, and baseline training setup. Use when: extracting notebook training logic, modifying the training loop, changing hyperparameters, or debugging loss behavior."
---

# Training Pipeline

## When to Use

- extracting notebook training code into src/
- modifying WeightedTrainer
- changing class weighting or focal loss behavior
- adjusting baseline training arguments

## Canonical Loss Pattern

```python
loss_per_sample = F.cross_entropy(
    logits, labels, weight=weights, reduction="none", label_smoothing=0.1
)
with torch.no_grad():
    probs = F.softmax(logits, dim=-1)
    pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
focal_term = (1 - pt) ** self.gamma
loss = (focal_term * loss_per_sample).mean()
```

## Common Failure Modes

- reduction=mean before focal scaling collapses the loss behavior
- TrainingArguments.label_smoothing_factor is ignored when compute_loss is overridden
- logging model_name to MLflow can conflict with autologged parameters; prefer base_model
- CamemBERTav2 LayerNorm remap warnings should be suppressed around model load and best-model reload

## Current Extraction Boundary

- ignore the notebook's last inference/demo cell on the ml branch
- keep notebook cells thin and reusable logic in src/