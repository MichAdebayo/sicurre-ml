# Training Plan

## Objective

Fine-tune a French email classifier for three classes: **phishing**, **spam**,
and **legitimate**, each recognised independently, rather than collapsing into
a binary phishing/not-phishing decision.

## Training path

1. Load a frozen dataset export produced by the companion repo and published to R2.
2. Validate schema and label mapping.
3. Tokenize for `almanach/camembertav2-base` at 256 tokens.
4. Train with `inverse_freq` class weighting, 4 epochs, batch size 8, LR 2e-5.
5. Evaluate the training/test splits and log to MLflow.
6. Register an immutable candidate and publish artifacts to Hugging Face.
7. Evaluate candidate and incumbent against the same immutable golden set.

Promotion is separate from training and subject to the evaluation gate and
configured human approval; see
[promotion policy](promotion-policy.md).

## Configuration that carries a rationale

**Epochs: 4, standardised.** This matches the budget the incumbent was trained
with. A three-epoch candidate on the 1 September corpus measured worse:
weighted F1 0.7141 against 0.8515, phishing recall identical at 0.8810, and
legitimate false positives rising from 8 to 20 (reported McNemar p = 0.0018).
That comparison does not isolate epoch count as the cause. A controlled
comparison is needed before making a causal claim. A perfect held-out score
from the same synthetic generators also does not prove saturation or
generalization to real inboxes.

**Class weighting: `inverse_freq`, with `phishing_boost` at 1.0.** Inverse
frequency equalizes nominal aggregate class weights, not information content,
per-example difficulty, or gradients throughout training. A subsequent
phishing boost changes relative loss weighting and remains a tunable parameter.
The standing value is neutral; any change needs independent validation.

**Corpus quality and coverage.** Reported corpus changes reduced the
phishing-to-legitimate ratio from roughly 1.88 to 1.23. Such a change may alter
coverage as well as balance; it does not by itself establish the cause of a
classification bias or prove its repair. Neither ratio is evidence of a
representative real-world inbox distribution. Dataset creation and review
remain owned by the companion Sicurre repository.

## Data assumptions

- Frozen datasets come from R2 and are the source of truth.
- Kaggle receives a packaged mirror of a chosen frozen version for execution.
- This repo never mutates raw or operational data.
- Corpus-shaping code changes and frozen dataset manifests are recorded by the
  companion repo before training. Reproduction also requires the corresponding
  immutable dataset artifacts, not source code alone.

## Known limitation

The evaluation split is drawn from the same generators as the training data, so
a high score on it is limited evidence of generalization. The separate
95-record golden set is a provisional regression gate, not a representative
customer benchmark. Statistically inconclusive comparisons do not establish
equivalence. The current measurements and their limitations are recorded in
[performance and quality](../architecture/performance.md).
