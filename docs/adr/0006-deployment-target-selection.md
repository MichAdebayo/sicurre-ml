# ADR-0006: Deployment Target Selection

**Date:** 2026-05-15
**Status:** Proposed

## Context

The broader Sicurre system may be deployed on Hetzner or Oracle Cloud. This ML repo does not host the user-facing application, but its operational runbooks need to reflect where companion services and automation may live.

## Decision

Keep deployment target selection open until the companion app hosting choice is finalized, while documenting assumptions that keep the ML pipeline portable.

## Alternatives considered

| Option | Pros | Cons |
|--------|------|------|
| Hetzner | Straightforward VM operations | Ongoing cost |
| Oracle Cloud free tier | Low cost for early deployment | More environment-specific constraints |
| Commit now before app decision is final | Faster documentation closure | Risk of documenting the wrong operational baseline |

## Consequences

Current docs should describe hosting assumptions at a high level only. Final deployment-specific runbooks should be completed after the companion app hosting decision is made.
