# ADR-0001: CamemBERTav2 As Base Model

**Date:** 2026-05-15
**Status:** Accepted

## Context

The current training notebook fine-tunes `almanach/camembertav2-base` for French email classification across phishing, spam, and legitimate classes. The repo needs a documented baseline before module extraction begins.

## Decision

Use CamemBERTav2 as the primary base model for the first structured extraction of the training pipeline.

## Alternatives considered

| Option | Pros | Cons |
|--------|------|------|
| CamemBERTav2 | French-oriented model already proven in the current notebook | Keeps the repo aligned to one backbone initially |
| Generic multilingual encoder | Broader language flexibility | Weaker alignment with the current French-first task |
| Smaller baseline model | Lower compute cost | Likely weaker task performance and less continuity with existing work |

## Consequences

The initial `src/` extraction should target the existing CamemBERTav2 training path first. Additional model families can be added later behind clear configuration boundaries.
