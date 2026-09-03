# ADR-0003: Kaggle As Training Runtime

**Date:** 2026-05-15
**Status:** Accepted

## Context

Training currently depends on Kaggle because the working notebook already runs there and benefits from available GPU capacity. At the same time, the data contract should stay outside Kaggle.

## Decision

Use Kaggle as the execution environment for managed training runs, while keeping R2 as the canonical store for frozen training datasets.

## Alternatives considered

| Option | Pros | Cons |
|--------|------|------|
| Kaggle runtime with R2-backed dataset lineage | Low-cost GPU access and clean data ownership boundary | Requires dataset packaging into Kaggle before training |
| Self-hosted GPU training | Full control | Higher operational cost and more infra scope |
| CPU-only local or CI training | Simpler infra | Not suitable for the intended training workload |

## Consequences

The source notebook remains in-repo, a Kaggle copy is pushed for execution, and the MLOps branch will own the automation that bridges frozen R2 exports to Kaggle Dataset versions.
