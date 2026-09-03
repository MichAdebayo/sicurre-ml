# Non-Functional Requirements

Engineering requirements, not claims of verified service levels. Measurements,
configured limits, and remaining verification are distinguished in
[performance](performance.md). These requirements do not constitute an SLA.

## Model quality

- Weighted F1, phishing recall, and legitimate false positives are all gated at
  promotion against the incumbent on the **same** immutable golden-set version.
- Phishing recall must not regress on that evaluation set; this does not prove
  non-regression across all real-world mail.
- False positives on legitimate email must remain visible in every evaluation
  summary because they directly affect users.
- The golden set is a promotion gate, not a benchmark. Runtime configuration
  must never be tuned against it.

## Reliability

- LLM provider unavailability alone should not prevent a verdict from the
  remaining stages. Authentication, validation, rate limiting, model failure,
  and unexpected exceptions can still produce non-success responses.
- Readiness must distinguish "model loaded" from "still downloading" so a slow
  start is never mistaken for a failure.
- Deployment is single-node without failover; objectives are set to be honest
  about that.

## Reproducibility

- Every training run should record one immutable frozen dataset identity.
- Every published model should resolve to one immutable artifact revision and
  its training source revision.
- Promotion decisions record the evaluated metrics, thresholds, and artifact
  references.
- Corpus changes belong to the Sicurre data platform and produce new frozen
  dataset identities. Code alone cannot recreate private training data; R2
  artifacts, manifests, checksums, and access permissions are also required.
- Missing runtime lineage must be reported, not guessed. The verified snapshot
  currently lacks its training dataset version; see
  [deployment identity](deployment-identity.md).

## Operability

- Local development supports dry runs on small fixtures.
- CI-triggered training is traceable to a run request and a dataset version.
- Failures in training orchestration must not affect the companion app runtime.
- Rollback must attempt to restore the preserved MLflow alias, Hugging Face tag,
  and deployed identity, then verify the result. A failed recovery must remain
  visible rather than being reported as successful.

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
