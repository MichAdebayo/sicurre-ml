# Monitoring design: model layer

## Scope

This repository monitors model training, evaluation, and the private inference
service. Email ingestion, database operations, forwarding, and user-facing
delivery belong to the companion `sicurre` repository. An inference dashboard
does not measure the complete email delivery path.

Measured results and their limitations belong in
[performance and quality](performance.md). Operational thresholds are not
customer SLAs or proof that performance objectives have been met.

## Current infrastructure

The production Compose configuration runs the app and an ML-owned Alloy
collector. Redis is not part of this architecture.

| Component | Role |
|-----------|------|
| `app` | Authenticated inference, readiness, structured logs, metrics, traces |
| `alloy` | Docker log collection, Prometheus scraping/remote write, OTLP processing/export |
| Grafana Cloud | Shared Prometheus, Loki, and Tempo backends; ML-scoped dashboard and alerts |

[docker-compose.prod.yml](../../docker-compose.prod.yml) pins Alloy to
`v1.16.1` and an immutable image digest.
[config.alloy](../../deploy/alloy/config.alloy) scrapes the app and Alloy every
60 seconds. Its secrets come from `deploy/env.alloy`.

The ML collector discovers only the `sicurre-ml` Compose project. Shared
datasources must remain additive: ML provisioning must not replace the
companion repository's dashboards or log views.

Use `stack="sicurre-ml"` to scope metrics and logs. The application has
`service_name="sicurre-ml-inference"`; collector self-observability uses
`service_name="sicurre-ml-alloy"`. Trace resources identify the inference
service through `service.name`.

## Signals and interpretation

| Signal | What it establishes | What it does not establish |
|--------|---------------------|----------------------------|
| Health, readiness, scrape `up` | Process/model state and scrape reachability | Customer email delivery success |
| Request count and rate | Observed inference traffic | Capacity or number of unique emails |
| Request latency by requested mode | HTTP processing time with LLM enabled or disabled | Email end-to-end time or isolated LLM latency |
| Per-stage latency | Time spent in ONNX, LLM, rules, or blocklist work | A cause for slowness without further diagnosis |
| Server errors and degraded outcomes | Explicit failures and fallback outcomes | Accuracy of successful classifications |
| Provider events and selection | Attempts, failures, fallback, selected provider | A confident or correct LLM answer |
| Label/verdict distribution | Composition of predictions | Ground-truth class distribution or model accuracy |
| Memory and Alloy delivery counters | Resource use and telemetry pipeline state | A complete host or billing audit |

A returned `uncertain` LLM label is different from provider unavailability.
A spam-heavy distribution can reflect traffic composition, missing subscription
context, or model bias. Inspect reviewed examples before drawing a conclusion.

ONNX class probabilities and the composite risk score are not interchangeable.
There is no general requirement that a healthy confidence distribution be
bimodal, and the current implementation should not be described as providing
a dedicated confidence histogram unless that panel and its data are verified.

The active LLM chain is Mistral then Groq. Its intended budget and the difference
between source defaults and deployed overrides are documented in
[performance](performance.md). Optional VirusTotal enrichment makes network
calls and should be analyzed separately from local blocklist lookups.

## Structured inference log

`emit_classify_request_log` in
[telemetry.py](../../src/serving/telemetry.py) emits these fields as applicable:

| Field | Meaning |
|-------|---------|
| `event` | `classify_request` |
| `status_code` | HTTP result |
| `latency_ms` | Scalar request-processing duration |
| `model_version` | Current caller supplies the loaded model SHA, despite this historical field name |
| `verdict` | Binary security verdict |
| `label_verdict` | Three-class classification |
| `label_distribution` | Bounded per-class distribution |
| `stage_latencies_ms` | Timing object keyed by pipeline stage |
| `llm_provider` | Selected provider or null |
| `error_type` | Bounded error category where applicable |

This is not the same schema as the identity manifest: there,
`model.version` is the readable version label and `model.revision` is the SHA.
See [deployment identity](deployment-identity.md).

The request log does not include the previously proposed `text_hash`,
`cache_hit`, raw email, prompt, or credential fields. Do not introduce PII or
unbounded request identifiers as metric labels.

Alloy drops repetitive successful probe access logs, not health metrics.
Failures and important lifecycle events remain diagnostically relevant.
Configured trace sampling is selective, so the absence of an individual
successful request in Traces Drilldown does not alone prove exporter failure.

## Dashboard interpretation

Keep the dashboard metrics-only. Logs and traces remain in their Drilldown
views. Keep the model version tag visible, with a short SHA for readability and
the full revision accessible through the manifest.

For sparse certification traffic:

- Show request counts alongside percentile and error-rate charts.
- Distinguish no requests, a genuine zero, and missing telemetry.
- Label percentiles as histogram estimates over the selected time window.
- Keep request latency separate from per-stage/provider latency.
- Distinguish LLM provider events from final provider selection and uncertainty.
- Do not present a verdict mix as a quality or accuracy score.

These are presentation requirements for the dashboard follow-up, not a claim
that every current panel already implements them. No dashboard layout was
changed during this documentation update.

## Alert definitions and known limitation

The repository-owned alert definitions are
[sicurre-ml-alerts.json](../../deploy/grafana/alerts/sicurre-ml-alerts.json).
They include readiness/availability, latency, server errors, authentication
and rate-limit spikes, memory pressure, scrape health, dropped logs, and ML
active-series budget warnings.

Both latency rules are now answerable. They were not: buckets ended at 5000 ms
while the LLM rule asked about 8000 ms, and `histogram_quantile` returns the
highest finite boundary for any quantile in the overflow bucket, so the query
could not produce a value exceeding the threshold however slow the service
became. Buckets were re-cut around the 2 s objective and the rule re-pointed at
2 s; a test binds the two so they cannot drift apart again. A configured rule
is still not evidence of successful firing, notification delivery, or on-call
coverage.

### Dropped log entries after a deploy

`loki_write_dropped_entries_total` climbed after each deploy — 55 rejections on
3 September — with Loki answering HTTP 400 "timestamp too old".

The cause was not the write-ahead log. Alloy had no `--storage.path`, so all of
its state, the WAL included, sat in the container's writable layer and was
destroyed by `--force-recreate` on every deploy. What actually replayed was
`loki.source.docker`: with its read positions gone it re-read the app's
retained json-file history, up to five files of 50 MB, and re-shipped entries
carrying their original timestamps. Loki rejected them as too old.

The distinction matters because the obvious fix — truncating the WAL on
restart — would have changed nothing; the WAL was already being destroyed. The
fix is a named volume at `/var/lib/alloy/data` with an explicit
`--storage.path`, so positions survive and Alloy resumes instead of replaying.
One further replay is expected on the first deploy after this change, since the
volume starts empty.

Evaluate window length and sample count when interpreting alerts. The
server-error expression clamps a small denominator, so at very low traffic it
is not simply an exact failed-requests percentage. The budget series query is
an instant count of ML-labeled series, not an independent Grafana billing
statement.

Samples per minute measure telemetry ingestion, not inference throughput and
not active-series count. A high sample-rate card must not reuse active-series
budget thresholds. Logs, traces, discarded telemetry, and testing usage have
separate budgets; visibility of one does not verify compliance with all of
them.

## Training and evaluation

MLflow retains training metrics, confusion matrices, candidate/incumbent
evaluations, and promotion evidence. Training-validation metrics are distinct
from the evaluation-only golden set and from production telemetry.

The promotion gate compares weighted F1, phishing recall, and legitimate false
positives against the incumbent on the same immutable set. There is no single
`eval/f1 >= promotion_tolerance` rule that substitutes for this comparison.
See [promotion policy](../model/promotion-policy.md).
