# ADR-0002: WeightedTrainer Focal Loss Design

**Date:** 2026-05-15
**Status:** Accepted

## Context

The current notebook relies on a custom trainer with class weighting and focal-style loss behavior to prioritize the phishing class and manage class imbalance.

## Decision

Preserve the custom `WeightedTrainer` approach during module extraction and treat its loss behavior as a first-class documented design choice.

## Alternatives considered

| Option | Pros | Cons |
|--------|------|------|
| Custom `WeightedTrainer` with focal loss | Matches the working notebook and supports explicit class-sensitive behavior | Requires careful tests during extraction |
| Stock Hugging Face `Trainer` only | Less custom code | Loses the task-specific loss behavior already being used |
| External training framework rewrite | Could standardize abstractions | Adds avoidable scope during certification work |

## Consequences

Module extraction should isolate trainer behavior into a dedicated module with focused tests. Any changes to loss weighting or promotion logic should update this ADR or a successor.
