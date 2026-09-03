# Performance and quality

## Scope of the evidence

The checks below are small, authenticated inference smoke tests. They are not a
load benchmark, an end-to-end email delivery measurement, or an SLA. A two-second
email classification objective has not been validated across the complete
delivery path by these checks.

The deployed mail path is:

```text
Cloudflare Email Worker -> sicurre /v1/email/scan -> sicurre-ml /v1/classify
```

The companion repository's Worker requests `use_llm=true` and
`use_virustotal=false`. The LLM-disabled path is useful for comparison but does
not represent the current Worker configuration.

## Verified inference checks, 2 September 2026

The timed run began at 18:51:29 UTC. Both running services reported model
revision `86e90dc5ff205a9711afe5de696f3d81ab8432be` and model version
`phishing-detector-1.0.0`. Local and production service/image identities differed;
see the [manifest snapshot and observations](evidence/inference-smoke-2026-09-02.json)
and [identity interpretation](deployment-identity.md).

Three synthetic French inputs were used in each environment and mode: a short
notice, a newsletter-style message, and a longer message, with text lengths of
237, 483, and 2,248 characters respectively. Inputs included unique references.
No real email content was used.

Requests were sequential, paced by about 1.1 seconds, with a persistent HTTP
client per environment, a 20-second client timeout, and no automatic retries.
Timing covers the authenticated HTTP classification round trip from the
operator's laptop, not model compute alone. The services were already running;
this was not a cold model-loading test. The first request per environment may
include connection establishment. VirusTotal enrichment was disabled throughout.

| Environment | Requested LLM | Requests | Median | Observed range |
|-------------|---------------|----------|--------|----------------|
| Local | Disabled | 3 | 151.24 ms | 64.12 to 190.21 ms |
| Local | Enabled | 3 | 900.95 ms | 829.30 to 1,073.15 ms |
| Production | Disabled | 3 | 489.76 ms | 462.12 to 634.18 ms |
| Production | Enabled | 3 | 864.72 ms | 758.12 to 1,135.51 ms |

All 12 requests returned HTTP 200. None returned 429 or a server error. Health
and readiness checks also returned 200. This small sample does not establish
p95/p99, worst-case latency, capacity, a population error rate, or failover
reliability. Differences between local and production medians cannot be
attributed to hardware or network latency from this sample alone.

### Compute is input-dependent

A separate, isolated local LLM-disabled request was measured against service
metric counter deltas. Exactly one classification was observed:

| Measurement | Time |
|-------------|------|
| Client HTTP round trip | 140.34 ms |
| Server request processing | 135.572 ms |
| ONNX stage | 134.657 ms |
| Rules stage | 0.101 ms |
| Blocklist stage | 0.088 ms |

These values do not support a general claim that classification compute takes
3 ms. Particular short inputs or fixtures can be much faster than other
requests.

### Availability is not classification quality

Mistral answered all six LLM-enabled requests without an
`llm_unavailable` degradation. However, the LLM stage returned `uncertain` in
all six. The final labels were legitimate for the short and long inputs and
spam for the newsletter-style input, matching the ONNX labels in these checks.

This verifies provider response availability for the tested requests. It does
not prove that the LLM contributed a confident second opinion, that a newsletter
was unsolicited, or that the classifications were correct. No accuracy score is
derived from these smoke tests.

## Corrections to the earlier measurements

The previously reported 393 ms `/v1/email/scan` median came from requests with
an invalid integration secret. Those requests were rejected with HTTP 401 after
authentication lookup, before successful-scan queries, inference, and
persistence. They did not exercise the complete scan.

The previously reported 426 ms `/v1/classify` measurement used
`use_llm=false`, despite being described as all four stages. Adding those two
medians does not measure the real end-to-end path, its median, or its worst
case. The earlier approximately 820 ms full-path estimate is withdrawn.

The attribution of 250 to 400 ms to network round-trip time was not established
by isolated measurements. The server was also incorrectly described as being
in Germany; the owner identifies the deployment location as Finland.

## Budgets, timeouts, and the latency alert

These numbers have different scopes:

| Setting | Production, from 3 September 2026 | Previously | Meaning |
|---------|-----------------------------------|-----------|---------|
| `LLM_TOTAL_TIMEOUT_SECONDS` | 1.5 s | 7.5 s | Budget for the whole provider chain |
| `LLM_PROVIDER_TIMEOUT_SECONDS` | 0.9 s | 4.5 s | Per-provider HTTP timeout |
| `LLM_CONNECT_TIMEOUT_SECONDS` | 0.3 s | 1 s | Connection establishment |
| `LLM_MAX_ATTEMPTS` | 1 | 1 | Attempts per provider — no retry multiplication |
| LLM-mode request p95 alert | 2 s | 8 s | Operational warning threshold |
| Worker scan-fetch timeout | 10 s | 10 s | Time the Worker waits for the scan fetch |

### How the budget was derived

The previous values were not chosen against the two-second objective; they
predate it. They were revised on 3 September 2026 by working backwards from it.

Five paced request pairs against production measured the LLM stage's marginal
cost: `use_llm=false` at a 366 ms median, `use_llm=true` at 823 ms, so the LLM
adds roughly **457 ms at the median and 714 ms at the observed worst**. Those
figures include about 200 ms of client round trip from a laptop; the Cloudflare
Worker sits at the edge and pays far less.

Fixed overhead outside the LLM — Worker to scan, the scan's own database work,
scan to ML, and the response — is estimated at about 315 ms. ONNX and the LLM
run in parallel, so the LLM budget replaces the ONNX cost rather than adding to
it. That leaves roughly 1,670 ms before two seconds is breached.

`0.9 s` per provider clears the observed worst with headroom while cutting off a
hung provider quickly. `1.5 s` for the chain means a first-provider timeout still
leaves 600 ms for the second — enough for a typical 457 ms response, so the
fallback stays useful rather than nominal. Worst case is about **1.83 s**.

These are budgets, not wall-clock guarantees; see the HTTPX note below.

Production was configured with `mistral-medium-2604` and
`openai/gpt-oss-120b` for the Mistral/Groq chain. Cerebras has an implementation
but is not in the active chain.

The intended 7.5-second budget is not a strict wall-clock cancellation guarantee.
The implementation checks the remaining budget between provider calls and
uses HTTPX timeouts for network operations. HTTPX distinguishes connect, read,
write, and pool timeouts; a read timeout limits waiting for a response chunk,
not the entire request duration. See the
[HTTPX timeout documentation](https://www.python-httpx.org/advanced/timeouts/).

The latency alert selects the whole `/v1/classify` request histogram for
`mode="llm"`. It does not isolate provider latency or prove that the LLM alone
exceeded its budget. The rule uses a 10-minute rate window and a 10-minute
pending period; it is not an immediate page for a single slow request.

### A monitoring defect, and its repair

Until 3 September 2026 this alert could not fire. The request histogram had
finite buckets only to 5,000 ms, and `histogram_quantile` returns the highest
finite boundary for any quantile landing in the overflow bucket — so p95 could
never report above 5,000 ms, while the rule asked whether it exceeded 8,000 ms.
The alert was green because the question was unanswerable, not because latency
was good. See
[Prometheus histogram_quantile semantics](https://prometheus.io/docs/prometheus/latest/querying/functions/#histogram_quantile).

The buckets were also thinnest where the objective sits: everything between one
and two seconds fell into a single bucket, so comfortably-inside and
about-to-breach were the same observation.

Boundaries are now `50, 100, 250, 500, 750, 1000, 1500, 2000, 3000` ms — five
below one second where healthy requests land, three across the band the
objective cares about, and a 3,000 ms ceiling that brackets what the configured
timeouts can now produce. The alert threshold moved to 2,000 ms, which is a real
bucket boundary rather than an interpolated estimate.

A test binds the two together: no latency alert may exceed the highest finite
boundary, and every threshold must land on one. Changing the objective now
requires changing both, or CI fails.

The Worker's 10-second timeout applies to its scan fetch. On scan failure it
uses a fail-open delivery path, but that timeout does not bound every forwarding
operation or prove cancellation of downstream inference. The companion
inference client also has its own 15-second timeout configuration.

For certification, describe an internal objective, a configured threshold, and
a measured result separately. These documents do not establish a contractual
SLA or claim that an objective has been met over a representative period.

## Model quality evidence

Existing evaluation records identify registry version 15 with the
`golden-20260816-v3` set:

| Metric | Recorded value |
|--------|----------------|
| Weighted F1 | 0.8515 |
| Phishing recall | 0.8810 |
| Legitimate false positives | 8 of 42 |

These evaluation results were not recomputed during the latency checks.
Registry version 15 is not the semantic model version returned by the API.

The promotion gate compares candidate and incumbent on the same immutable
evaluation set: weighted F1 and phishing recall must not fall, and legitimate
false positives must not increase. Human review remains necessary. See the
[promotion policy](../model/promotion-policy.md).

The 95-record provisional set is useful for regression decisions, not a
representative French customer benchmark. A statistically inconclusive
comparison does not prove equivalence, and a numerical gate pass does not
establish real-world quality. Repeatedly tuning against the golden set would
also weaken its value as an independent check.

## Remaining verification

- Measure a successful authorized Worker-to-scan-to-inference path, including
  persistence and forwarding, with a controlled test integration.
- Retain a repeatable harness and sanitized fixture definitions for future
  benchmarks. The current evidence file preserves observations and method,
  not an exact replay of every generated input.
- Measure varied input lengths, cold model startup, controlled concurrency,
  sustained traffic, provider failures, rate limiting, and recovery separately.
- Verify the repaired latency alert end to end: confirm it fires on a
  deliberately slow request and that the notification arrives, now that the
  threshold sits inside the histogram's range and can be reached.
- Recover the frozen training dataset identity for the current production
  artifact. The observed runtime manifest still reports `dataset.version` as
  `unknown`; do not replace it with a model SHA or golden-set version.

No training, promotion, deployment, or production configuration change was
performed for this documentation update.
