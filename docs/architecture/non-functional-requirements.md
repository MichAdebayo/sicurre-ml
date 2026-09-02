# Non-Functional Requirements

Qualities the system must hold. **Numeric targets are not restated here** —
they live in [service levels](service-levels.md) and are referenced from it, so
there is one place to change a number and no chance of two documents disagreeing.

## Model quality

- Weighted F1, phishing recall, and legitimate false positives are all gated at
  promotion against the incumbent on the **same** golden-set version (`M1`–`M3`).
- Phishing recall is the primary safety metric and must never regress.
- False positives on legitimate email must remain visible in every evaluation
  summary — they are the failure mode users actually feel.
- The golden set is a promotion gate, not a benchmark. Runtime configuration
  must never be tuned against it.

## Reliability

- The service must return a verdict whenever it is ready, even when the LLM
  provider chain is unavailable — degradation is a quality event, not an
  outage (`S2`, `S5`).
- Readiness must distinguish "model loaded" from "still downloading" so a slow
  start is never mistaken for a failure.
- Deployment is single-node without failover; objectives are set to be honest
  about that.

## Reproducibility

- Every training run resolves to one specific dataset version.
- Every published model resolves to one specific source revision.
- Promotion decisions record the evaluated metrics, thresholds, and artifact
  references (`P2`, `P3`).
- Any change to corpus composition must be committed to source before the run
  that consumes it, so the dataset can be rebuilt from the repository alone.

## Operability

- Local development supports dry runs on small fixtures.
- CI-triggered training is traceable to a run request and a dataset version.
- Failures in training orchestration must not affect the companion app runtime.
- Rollback restores every preserved pointer — MLflow alias, Hugging Face tag,
  and deployed identity — and is verified, not assumed (`P4`).

## Security

- Secrets come from environment-specific secret stores, never committed files.
- Training artifacts must not expose raw private data in logs or reports.
- No secret, raw email content, generated sample text, or user PII belongs in a
  promotion manifest or MLflow tag.
- Inference endpoints that expose model identity or accept text require a
  bearer token; rate limiting is on by default.

## Documentation

- A numeric target appears in exactly one document.
- A decision with a non-obvious rationale carries that rationale where the
  decision lives, not only in a commit message.
