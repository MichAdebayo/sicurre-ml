# Service Levels — SLI, SLO, SLA

## Why this document exists

Model quality and service reliability were previously described only as
adjectives: "should meet the current promotion threshold", "target: <500 ms".
Adjectives cannot be measured, alerted on, or put on a dashboard, and they give
anyone joining the project no way to tell whether the system is healthy.

This document is the single source of numeric truth for what `sicurre-ml`
promises. Every dashboard panel, alert rule, and promotion gate should trace
back to a row here rather than reintroducing its own thresholds.

## The three terms, used precisely

They are routinely conflated, so this repo fixes their meanings:

| Term    | What it is | Audience | Consequence of breach |
|---------|------------|----------|-----------------------|
| **SLI** | A measurement. A number the system actually emits. | Engineers | None — it is just a reading |
| **SLO** | An internal target for an SLI, with an error budget. | The team | Work stops on features; reliability takes priority |
| **SLA** | An external promise to a consumer. | The `sicurre` app | Documented degradation path and incident record |

The rule of thumb this repo follows: **the SLA is looser than the SLO**. The
gap is deliberate headroom, so the internal target breaks first and gives
warning before an external promise is broken.

## Parties

`sicurre-ml` is the **provider**. It exposes a classification API and publishes
model artifacts.

The `sicurre` application is the **consumer**. It calls `/v1/classify` and acts
on the verdict.

There is no third-party or paying customer, so the SLA carries no financial
remedy. It is an engineering commitment between two repos owned by the same
maintainer, and is written to be defensible rather than contractual.

---

## Plane 1 — Serving (the running inference API)

The deployment is a **single Hetzner CX33 node** running Docker Compose. There
is no redundancy, no load balancer, and no failover. Every number below is
chosen to be honest about that rather than to look impressive: a 99.9% promise
on one box without HA would be fiction.

### SLIs

| ID | SLI | How it is measured |
|----|-----|--------------------|
| `S1` | Readiness | `/v1/ready` returns 200 (model loaded) rather than 503 (still downloading) |
| `S2` | Request success | Share of `/v1/classify` calls returning 2xx rather than 5xx |
| `S3` | ONNX stage latency | `latency_ms.onnx` from the structured request log |
| `S4` | Full-path latency | `latency_ms.total` — gated by the LLM stage when enabled |
| `S5` | Degraded-mode rate | Share of responses carrying a `degraded_reasons` entry |

### SLOs

| ID | Objective | Target | Window | Error budget |
|----|-----------|--------|--------|--------------|
| `S1` | Readiness | **99.0%** | 30 days | 7 h 12 m |
| `S2` | Request success | **99.0%** | 30 days | 1 in 100 calls |
| `S3` | ONNX p95 | **< 500 ms** | 7 days | — |
| `S4` | Full-path p95 | **< 8 s** (LLM enabled) · **< 1 s** (degraded) | 7 days | — |
| `S5` | Degraded-mode rate | **< 10%** | 7 days | — |

### SLA offered to `sicurre`

| Promise | Value |
|---------|-------|
| Readiness | **98.5%** monthly |
| A verdict is always returned when the service is ready | **100%** — see graceful degradation |
| Full-path response | **< 20 s**, after which the caller should time out |
| Planned-maintenance exclusion | Deploys and model swaps, announced in advance |

### Graceful degradation is the load-bearing property

The pipeline runs four stages — `rules`, `blocklist`, `onnx`, `llm`. The LLM
stage is dispatched to a thread pool and joined before the composite score is
formed, so it gates total latency; but when the whole provider chain fails, the
pipeline records `llm_unavailable` in `degraded_reasons` and **still returns a
verdict** from the remaining stages.

This is why `S2` (a verdict was returned) and `S5` (it was a full-confidence
verdict) are tracked separately. An LLM outage should show up as a degraded-mode
spike, never as an availability incident. Collapsing the two would hide the
single most likely failure in the system — a provider outage or an expired API
key — behind a metric that still looks green.

---

## Plane 2 — Model quality

These are **gates**, not continuously-served objectives: they are evaluated at
promotion time against an immutable golden set, not per request.

### The incumbent

Production is model **v15** (Hugging Face tag `86e90dc5`), measured on
`golden-20260816-v3`:

| Metric | Value |
|--------|-------|
| Weighted F1 | **0.8515** |
| Phishing recall | **0.8810** |
| Legitimate false positives | **8** of 42 |

### Promotion SLOs

A candidate may only replace the incumbent if **all** hold on the *same*
golden-set version:

| ID | Gate | Rule |
|----|------|------|
| `M1` | Weighted F1 | ≥ incumbent |
| `M2` | Phishing recall | ≥ incumbent — never regress detection |
| `M3` | Legitimate false positives | ≤ incumbent |
| `M4` | Label and response contract | Byte-identical shape |
| `M5` | Human approval | Protected `production` environment review |

### Two cautions that have already cost real time

**Never compare across golden-set versions.** A candidate scoring 0.8401 on
`v3` was once reported as a pass against an incumbent's 0.7965 on `v1`. On the
same set the incumbent scored 0.8515 and the candidate lost on both. Any
comparison must name one golden-set version for both models.

**The golden set is a promotion gate, not a benchmark.** It is 95 samples.
It is small enough that most differences between candidates are noise — in one
series of eight retrains, every McNemar test but one returned p = 1.0. Runtime
configuration must never be tuned against it, or the gate stops being an
independent check and becomes a training target.

---

## Plane 3 — Pipeline and lineage

| ID | SLI | Objective |
|----|-----|-----------|
| `P1` | Dataset freshness | A release builds monthly, on the 3rd |
| `P2` | Lineage completeness | **100%** of promotions carry dataset ID, version, checksum, MLflow run, HF SHA |
| `P3` | Reproducibility | Every published model resolves to one source revision and one dataset version |
| `P4` | Rollback time | **< 15 min** from decision to verified incumbent restored |

`P2` and `P4` are not aspirational: the promotion workflow preserves the
previous MLflow alias, HF tag, and deployed identity before touching anything,
and restores all three with a `rolled_back` callback if any post-approval step
fails.

---

## Alert rules derived from these objectives

Alerts fire **before** an SLO is breached, not after.

| Condition | Threshold | Relates to | Action |
|-----------|-----------|------------|--------|
| `/v1/ready` non-200 | > 2 consecutive min | `S1` | Page |
| 5xx rate | > 5% over 5 min | `S2` | Page |
| ONNX p95 | > 2 s | `S3` | Investigate memory/CPU |
| LLM fallback rate | > 50% over 15 min | `S5` | Check provider keys |
| Verdict phishing rate | > 30% shift from 7-day baseline | Model drift | Investigate population change |

## What is deliberately not covered

Ingestion health, watch renewals, remediation outcomes, notification delivery,
and database metrics belong to the `sicurre` application repo. This document
stops at the classification API boundary.

## Review cadence

These numbers are provisional until the structured request log is emitting in
production and has thirty days of history. Once it does, every SLO here should
be re-derived from observed distributions rather than from targets set in
advance, and this section replaced with the date it was last calibrated.
