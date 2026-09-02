# Monitoring Design — Model Layer

## Scope

This document defines the monitoring strategy for the `sicurre-ml` repository.

It covers the ML training pipeline and the ONNX inference service only.
Application-level concerns (message ingestion, remediation workflows, user
notifications, watch renewals) belong to the `sicurre` application repo and are
explicitly out of scope here.

---

## Why monitoring is a delivery bloc

Monitoring is not a post-launch concern.
It provides certification-visible evidence of:

- model quality awareness during training
- operational control over the inference service
- early detection of degradation before user impact
- traceable incident response linked to observable data

---

## What this repo monitors

### 1. Model availability

- Is the ONNX Runtime session loaded and responsive?
- Which model SHA is currently active? (logged at startup from the SHA cache file)
- Exposed by `/v1/ready`: returns 200 when the model is loaded, 503 while it is
  still downloading.

### 2. Per-stage inference latency

Each of the four pipeline stages is timed independently:

| Stage      | Why it matters |
|------------|----------------|
| `rules`    | Should be sub-millisecond; a spike indicates a regex change |
| `blocklist`| Should be sub-millisecond; a spike indicates a slow TLD lookup |
| `onnx`     | Target: <500 ms; a spike indicates memory pressure or session reload |
| `llm`      | Target: <20 s; tracks Groq and Cerebras response times separately |

### 3. Verdict distribution

The fraction of requests classified as `phishing` vs `safe` is logged per request
and tracked over time. A sudden shift (e.g., sustained >80% phishing) is an
early signal that either the traffic population or the model behaviour has changed.

### 4. ONNX confidence distribution

Healthy model output is bimodal: most composite scores cluster near 0 or near 1.
If the distribution flattens toward 0.5 (the threshold), the model is becoming
uncertain. This is logged as the raw `composite_score` per request and visualised
as a histogram in Grafana.

### 5. LLM tier usage and fallback rate

- Which LLM provider was used (`groq`, `cerebras`, or `none`)?
- How often does Tier 1 (Groq) fail and fall back to Tier 2 (Cerebras)?
- How often does the whole LLM stage return `None` (both providers failed)?

A spike in fallback rate signals API key expiry or a provider outage.

### 6. Prediction cache hit rate

Once the Redis prediction cache is active, the fraction of requests served from
cache vs. computed fresh is logged. A low hit rate with repetitive inputs may
indicate a cache configuration issue. A high hit rate reduces LLM costs
directly.

### 7. Rate limit hit rate

How often is a caller rejected by the rate limiter (HTTP 429)?
A sustained spike may indicate a misconfigured client, a stuck retry loop, or
an attempted abuse.

### 8. Training quality signals (mlops branch)

Tracked via MLflow during training runs:

- `eval/f1` per class and weighted
- `eval/precision`, `eval/recall`
- `eval/loss`
- Confusion matrix (as a logged artifact)
- Promotion threshold passage (`eval/f1 >= promotion_tolerance`)

These are logged per training run, not per inference request.

---

## What this repo does NOT monitor

The following concerns belong to the `sicurre` application repo:

- message ingestion pipeline health
- watch renewal and listener availability
- remediation workflow outcomes
- user-facing notification delivery
- database metrics
- user session data

---

## Infrastructure

Sicurre-ML runs its own isolated monitoring stack — not shared with Vinse.
Both applications run on the same Hetzner server (CX33) but in separate
Docker Compose projects under separate Linux users.

| Component       | Image                  | Role |
|-----------------|------------------------|------|
| `app`           | sicurre-ml (GHCR)      | FastAPI inference API |
| `redis`         | redis:7-alpine         | Rate limiting + prediction cache |
| `alloy`         | grafana/alloy:latest   | Log shipping agent |

Grafana Alloy tails the `app` container's stdout and ships logs to Grafana Cloud
Loki. No metrics agent (Prometheus) is deployed at this stage.

---

## Log schema

Target state for model-runtime telemetry in this service: each completed
`/v1/classify` request should emit one structured JSON log line to stdout.
Planned fields:

| Field              | Type    | Description |
|--------------------|---------|-------------|
| `request_id`       | string  | UUID, unique per request |
| `text_hash`        | string  | SHA-256 of input text (no plaintext logged) |
| `verdict`          | string  | `phishing` or `safe` |
| `composite_score`  | float   | Final weighted score (0–1) |
| `stage_scores`     | object  | `{rules, blocklist, onnx, llm}` individual scores |
| `latency_ms`       | object  | `{rules, blocklist, onnx, llm, total}` in milliseconds |
| `llm_provider`     | string  | `groq`, `cerebras`, or `null` |
| `cache_hit`        | bool    | Whether the result was served from Redis |
| `model_sha`        | string  | SHA of the loaded ONNX model file |
| `timestamp`        | string  | ISO 8601 UTC |

---

## Alerting targets

| Condition | Threshold | Action |
|-----------|-----------|--------|
| `/v1/ready` returns non-200 | > 2 consecutive minutes | Page on-call |
| API error rate (5xx) | > 5% over 5 minutes | Page on-call |
| LLM fallback rate | > 50% over 15 minutes | Alert — check API keys |
| Verdict phishing rate | > 30% shift from 7-day baseline | Alert — check model drift |
| ONNX latency p95 | > 2 s | Alert — check memory/CPU |

---

## Delivery outputs

Expected outputs for the monitoring delivery bloc:

- this design document (public, certification-visible)
- `deploy/alloy/config.alloy` — Alloy configuration wired to Grafana Cloud
- structured JSON logging in `src/serving/app.py` and `src/inference/pipeline.py` (not yet implemented)
- Grafana dashboard capturing verdict distribution, latency, and fallback rate
- one documented incident example with root cause, fix, and linked evidence
