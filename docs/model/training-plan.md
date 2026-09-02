# Training Plan

## Objective

Fine-tune a French email classifier for three classes: **phishing**, **spam**,
and **legitimate** — each recognised independently, rather than collapsing into
a binary phishing/not-phishing decision.

## Training path

1. Load a frozen dataset export produced by the companion repo and published to R2.
2. Validate schema and label mapping.
3. Tokenize for `almanach/camembertav2-base` at 256 tokens.
4. Train with `inverse_freq` class weighting, 4 epochs, batch size 8, LR 2e-5.
5. Evaluate against the current immutable golden set and log to MLflow.
6. Register an immutable candidate; publish artifacts to Hugging Face.

Promotion is a separate operation and never happens automatically — see
[promotion policy](promotion-policy.md).

## Configuration that carries a rationale

**Epochs: 4, standardised.** This matches the budget the incumbent was trained
with, so candidates are comparable rather than systematically undertrained.
Three was tried on the 1 September corpus and measured significantly worse:
weighted F1 0.7141 against 0.8515, phishing recall identical at 0.8810, and
legitimate false positives rising from 8 to 20 (McNemar p = 0.0018). Detection
was unchanged and discrimination collapsed. The earlier argument for three —
that a 1.00 score on the held-out split meant the model had saturated — was
wrong: that split is drawn from the same generators as train, so 1.00 measures
how easy the split is, not how finished the learning is.

**Class weighting: `inverse_freq`, with `phishing_boost` at 1.0.** Inverse
frequency makes `n x w` constant across classes, so it cancels the imbalance
exactly. A boost applied afterwards therefore survives that cancellation as an
invariant multiplier on the phishing class rather than as a tunable — which is
why it sits at 1.0 rather than being used to chase recall.

**Corpus balance over loss weighting.** When the corpus ran at 1.88x phishing to
legitimate, the failure was not gradient imbalance — `inverse_freq` already
handles that — but information: the model saw nearly twice as many distinct
phishing emails and learned a thinner legitimate concept. The fix was to
generate more legitimate mail, bringing the ratio to roughly 1.23x. Deliberately
not parity, since real inboxes are not balanced either.

## Data assumptions

- Frozen datasets come from R2 and are the source of truth.
- Kaggle receives a packaged mirror of a chosen frozen version for execution.
- This repo never mutates raw or operational data.
- Corpus-shaping changes are committed to the companion repo before the run that
  consumes them, so lineage is reproducible from source.

## Known limitation

The evaluation split is drawn from the same generators as the training data, so
a high score on it measures split difficulty rather than generalisation. The
golden set exists precisely because of this, and it is small (95 samples) —
large enough to gate a promotion, too small to rank close candidates. Most
differences between candidates in one eight-retrain series returned McNemar
p = 1.0.
