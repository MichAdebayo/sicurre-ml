# ADR-0004: MLflow For Experiment Tracking

**Date:** 2026-05-15
**Status:** Accepted

## Context

The notebook already integrates with MLflow. The extracted training pipeline needs a durable way to log runs, metrics, parameters, and artifact references across local and remote execution contexts.

## Decision

Use MLflow as the experiment tracking layer for this repo, with support for a local fallback when the preferred remote backend is unavailable.

## Alternatives considered

| Option | Pros | Cons |
|--------|------|------|
| MLflow | Already integrated and matches the training workflow | Requires environment-specific backend configuration |
| Ad hoc file logging | Minimal setup | Poor run comparison and weak artifact lineage |
| Replace with another tracking system now | Could simplify one environment | Unnecessary migration during extraction phase |

## Consequences

Training modules should log through a stable MLflow abstraction. Remote registry details can evolve by environment, but the repo should not depend on notebook-only logging behavior.
