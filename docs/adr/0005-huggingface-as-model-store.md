# ADR-0005: Hugging Face As Production Model Store

**Date:** 2026-05-15
**Status:** Accepted

## Context

The main Sicurre app needs a deterministic way to consume trained model artifacts without coupling itself to the training environment.

## Decision

Publish approved model artifacts to Hugging Face and have the companion app pin explicit model revisions.

## Alternatives considered

| Option | Pros | Cons |
|--------|------|------|
| Hugging Face with pinned revision | Clear artifact boundary and easy app consumption | Requires revision management discipline |
| Direct artifact handoff between repos | Simple at first | Tight coupling and weak release traceability |
| Serve directly from training environment | No extra publication step | Couples inference to training runtime |

## Consequences

Promotion workflows must capture the exact published revision. The app repo should update pinned model references explicitly rather than depending on an implicit latest artifact.
