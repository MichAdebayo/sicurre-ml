# Service Levels — SLI, SLO, SLA

## Why this document exists

Model quality and service reliability were previously described only as
adjectives: "should meet the current promotion threshold", "target: <500 ms".
Adjectives cannot be measured, alerted on, or put on a dashboard.

This document is the single source of numeric truth for `sicurre-ml`. Every
dashboard panel, alert rule, and promotion gate should trace back to a row here
rather than carrying its own threshold.

## The three terms, used precisely

They are routinely conflated. Following
[Google SRE](https://sre.google/sre-book/service-level-objectives/):

| Term | What it is | Audience | On breach |
|------|------------|----------|-----------|
| **SLI** | A measurement — a ratio the system actually emits | Engineers | Nothing; it is a reading |
| **SLO** | A target for an SLI over a stated window, with an error budget | The team | Reliability work takes priority over features |
| **SLA** | A commitment to a consumer, **with agreed consequences** | The `sicurre` app | The consequences below are triggered |

An SLA is not a third colour threshold. If a number has no consequence attached
it is an SLO, not an SLA.

The SLA is deliberately looser than the SLO, so the internal target breaks first
and gives warning before a commitment is broken.

## Measurement status

The service **is instrumented and collecting**. `/v1/classify` emits a
structured log line per request via `emit_classify_request_log`
(`src/serving/app.py`) carrying `latency_ms`, `stage_latencies_ms`, `verdict`,
`llm_provider`, and `model_version`. Grafana Alloy ships those to Grafana Cloud,
and the runtime dashboard renders p95 latency, error rate, degraded decisions,
and provider usage.

What is **not** established is whether the observed traffic is representative
and whether the targets below are actually attained over a full window. The
numbers here were set from the deployed alert thresholds and the timeouts the
code enforces — not derived from thirty days of production distribution. Treat
them as the definition of what to hold, pending calibration.

## Parties

`sicurre-ml` is the **provider**; the `sicurre` application is the **consumer**.
Both are owned by the same maintainer, so the SLA carries no financial remedy —
its consequences are operational and are listed explicitly below.

---

## Plane 1 — Serving

Deployment is a **single Hetzner CX33 node** running Docker Compose. No
redundancy, no load balancer, no failover. The targets are chosen to be honest
about that: a 99.9% promise on one box would be fiction.

### Two request modes, which must not be conflated

`/v1/classify` takes `use_llm` (default `true`), recorded on each log line as
`mode`:

| Mode | Stages | Bound |
|------|--------|-------|
| **`local`** | `rules`, `blocklist`, `onnx` | No network LLM call. Sub-second |
| **`llm`** | the above plus the LLM chain | LLM chain hard-bounded by `LLM_TOTAL_TIMEOUT_SECONDS` (default **7.5 s**), each provider by `LLM_PROVIDER_TIMEOUT_SECONDS` (default **2.5 s**) |

**Degraded is a third thing, and is not the same as `local`.** A degraded
response is a `mode=llm` request whose provider chain returned nothing: the
pipeline records `llm_unavailable` in `degraded_reasons` and scores from the
remaining stages. It has *already spent up to the 7.5 s deadline waiting*, so it
is slow and low-confidence — the opposite of `local`, which is fast by choice.
Reporting the two together would hide provider outages inside a healthy-looking
latency figure.

The provider chain is `_TIERS = (_call_mistral, _call_groq)` —
**Mistral, then Groq**, sequential within one shared deadline. Cerebras is
implemented but is not in the chain.

### SLIs

| ID | SLI | Source |
|----|-----|--------|
| `S1` | Readiness: `/v1/ready` returns 200 rather than 503 | Probe |
| `S2` | Success ratio: share of `/v1/classify` returning 2xx | Structured log |
| `S3` | `mode=local` p95 of `latency_ms` | Structured log |
| `S4` | `mode=llm` p95 of `latency_ms` | Structured log |
| `S5` | Degraded ratio: share of `mode=llm` responses carrying `degraded_reasons` | Structured log |

### SLOs

| ID | Objective | Target | Window | Error budget |
|----|-----------|--------|--------|--------------|
| `S1` | Readiness | 99.0% | 30 d | 7 h 12 m |
| `S2` | Success ratio | 99.0% | 30 d | 1 in 100 requests |
| `S3` | `mode=local` p95 | < 1 s | 7 d | — |
| `S4` | `mode=llm` p95 | < 8 s | 7 d | — |
| `S5` | Degraded ratio | < 10% | 7 d | — |

`S3` and `S4` are set equal to the deployed alert thresholds so a page and a
breach mean the same thing. `S4`'s 8 s sits just above the 7.5 s chain deadline,
leaving room for the local stages and serialisation.

### SLA offered to `sicurre`

| Commitment | Value |
|------------|-------|
| Readiness | 98.5% monthly |
| A verdict is returned whenever the service is ready | 100% — guaranteed by degradation, never by the LLM |
| Bounded response | The LLM chain cannot exceed its 7.5 s deadline; callers should time out at 10 s |
| Notice of planned maintenance | Deploys and model swaps announced in advance |

**Consequences on breach** — this is what makes the above an SLA rather than
another threshold:

1. An incident record is opened with root cause and linked evidence.
2. **Model promotion is frozen** until the service has been inside `S1` and `S2`
   for seven consecutive days.
3. If the breach followed a promotion, the incumbent is restored using the
   preserved pointers before anything else is attempted.
4. The affected objective is re-derived from observed data rather than
   reasserted.

---

## Plane 2 — Model quality

Gates evaluated at promotion time against an immutable golden set, not per
request.

### The incumbent

Production is **v15** (Hugging Face tag `86e90dc5`) on `golden-20260816-v3`:

| Metric | Value |
|--------|-------|
| Weighted F1 | 0.8515 |
| Phishing recall | 0.8810 |
| Legitimate false positives | 8 of 42 |

### Promotion gates

All must hold on the **same** golden-set version:

| ID | Gate | Rule |
|----|------|------|
| `M1` | Weighted F1 | ≥ incumbent |
| `M2` | Phishing recall | ≥ incumbent — never regress detection |
| `M3` | Legitimate false positives | ≤ incumbent |
| `M4` | Label and response contract | Identical shape |
| `M5` | Human approval | Protected `production` environment |

### Two cautions that have already cost time

**Never compare across golden-set versions.** A candidate scoring 0.8401 on
`v3` was once called a pass against an incumbent's 0.7965 on `v1`. On the same
set the incumbent scored 0.8515 and the candidate lost on both.

**The golden set is a gate, not a benchmark.** At 95 samples most differences
between candidates are noise — in one series of eight retrains every McNemar
test but one returned p = 1.0. Runtime configuration must never be tuned against
it, or it stops being an independent check.

---

## Plane 3 — Pipeline and lineage

| ID | Objective | Target |
|----|-----------|--------|
| `P1` | Dataset freshness — a release builds monthly | 3rd of the month |
| `P2` | Lineage completeness on promotions | 100% |
| `P3` | Reproducibility: one source revision, one dataset version per model | 100% |
| `P4` | Rollback time from decision to verified incumbent | < 15 min |

---

## Deployed alerts

These exist in `deploy/grafana/alerts/sicurre-ml-alerts.json` today.

| Alert | Threshold | Relates to |
|-------|-----------|------------|
| Service unavailable | — | `S1` |
| Model not ready | — | `S1` |
| Local inference p95 above 1 s | 1000 ms | `S3` |
| LLM inference p95 above 8 s | 8000 ms | `S4` |
| Server error rate above 2% | 0.02 | `S2` |
| Authentication rejection spike | 5 | Security |
| Rate-limit spike | 10 | Abuse or client misconfiguration |
| Process memory above 6 GiB | 6 GiB | Capacity |
| Telemetry scrape unavailable | — | Observability itself |
| Alloy dropping log entries | 0 | Observability itself |
| Active series above 70% / 85% budget | 2100 / 2550 | Cost |

### Known gap

The error-rate alert fires at **2%** while `S2` budgets **1%**. The page
therefore arrives after the objective is already spent, not before. The correct
fix is a multi-window burn-rate alert on the `S2` budget rather than a static
rate; until that exists, treat the 2% page as confirmation of a breach rather
than warning of one.

## Out of scope

Ingestion health, watch renewals, remediation outcomes, notification delivery,
and database metrics belong to the `sicurre` application repo. This document
stops at the classification API boundary.

## Calibration

Every serving number here should be re-derived from observed distributions once
a full window of production traffic exists, and this section replaced with the
date it was last calibrated. Not yet calibrated.
