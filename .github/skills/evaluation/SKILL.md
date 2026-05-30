---
name: evaluation
description: "Evaluation, confusion matrix, error analysis, and promotion thresholds for the Sicurre classifier. Use when: modifying evaluation metrics, debugging test outputs, or checking promotion criteria."
---

# Evaluation

## When to Use

- adjusting metrics returned to the trainer
- debugging class-specific precision, recall, or false positives
- fixing notebook evaluation ordering issues

## Promotion Gate

```python
RECALL_THRESHOLD = 0.97
F1_THRESHOLD = 0.90
if phishing_recall >= RECALL_THRESHOLD and f1_weighted >= F1_THRESHOLD:
    production = True
else:
    production = False
```

## Safety Metrics

- phishing_recall
- phishing_fp_rate
- class-level precision, recall, and F1 for all three labels

## Notebook Ordering Rule

Generate predictions before building error-analysis artifacts. The confusion-matrix/prediction step must happen before any cell that depends on pred_labels or prediction probabilities.