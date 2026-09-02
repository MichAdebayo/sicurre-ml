# Performance and quality

## The bar

**Two seconds, end to end.** That is the number Sicurre is trying to beat for a
classification, because mail should not feel held up.

Everything below is either a measurement or context that explains one. None of
it is a dashboard requirement.

## What was measured

Measured 2 September 2026 against the production model (v15, Hugging Face
revision `86e90dc5`), classifying French email.

### The full path

An email reaches a verdict through three hops:

```
Cloudflare Email Worker  ->  sicurre  /v1/email/scan  ->  sicurre-ml  /v1/classify
```

| Hop | Measured | How |
|-----|----------|-----|
| `/v1/email/scan` — network, app, database lookup | **393 ms** median (255–473) | 8 requests from a laptop over the public internet |
| `/v1/classify` — network, plus all four pipeline stages | **426 ms** median (269–439) | 10 requests from a laptop over the public internet |
| Classification compute alone, no network | **3 ms** median, 36 ms p95 | 15 unique inputs against a local server |
| `rules` stage alone | 0.001 ms median | 600 in-process calls |
| `blocklist` stage alone, local list | 0.001 ms median | 600 in-process calls |

**The full path is comfortably inside two seconds.** Even adding the two
internet-measured hops naively — 393 + 426 ≈ 820 ms — leaves more than half the
budget unused, and that is the pessimistic reading.

It is pessimistic because both figures were taken from a laptop in France to a
server in Germany, where roughly 250–400 ms of each is simply network round
trip. The real path does not look like that: the Cloudflare Worker runs at the
edge, much closer to the server, and `/v1/email/scan` reaches `/v1/classify`
across the same machine rather than the public internet. The work itself is
3 milliseconds.

So two seconds is not a number to be nervous about. It is roughly four times
the measured worst case from the worst realistic vantage point.

### The one thing that does not fit

The LLM stage, when enabled, is allowed 7.5 seconds
(`LLM_TOTAL_TIMEOUT_SECONDS`, 2.5 s per provider, Mistral then Groq). That does
not fit a two-second budget and is not meant to — it is the slower, more careful
second opinion. Requests that must be fast run with `use_llm=false`, which is
the path measured above.

### Where the number 8 comes from

It appears in the Grafana alerts as *"LLM inference p95 above 8s"*. It is an
alarm threshold, not a promise and not a timeout: it sits just above the LLM's
own 7.5-second budget so that a page fires when the chain overruns. For
completeness, the related numbers are:

| Value | What it is |
|-------|------------|
| 7.5 s | What the code allows the LLM chain |
| 8 s | The alert that fires when the LLM p95 exceeds that |
| 10 s | When the Cloudflare Worker stops waiting and delivers anyway |

None of these describe the ONNX path, which is what the two-second figure covers.

### Method

Unique text on every request so nothing could be served from a cache, and the
loaded model confirmed through `/v1/manifest` before timing. All timing ran from
throwaway scripts outside the repository — no source was changed to produce a
number.

## Model quality

Production is **v15** on the `golden-20260816-v3` evaluation set:

| Metric | Value |
|--------|-------|
| Weighted F1 | 0.8515 |
| Phishing recall | 0.8810 |
| Legitimate false positives | 8 of 42 |

A new model replaces it only if, **on the same evaluation set**, it scores at
least as well on all three: weighted F1, phishing recall, and legitimate false
positives — and a human approves it.

### Two things that have already caught us out

**Compare on the same evaluation set.** A candidate scoring 0.8401 on `v3` was
once called a pass against an incumbent's 0.7965 on `v1`. On the same set the
incumbent scored 0.8515 and the candidate lost on both.

**The evaluation set is a gate, not a benchmark.** At 95 samples, most
differences between candidates are noise — across eight retrains, every
statistical test but one came back inconclusive. It decides whether a model may
ship; it does not rank close models, and nothing should be tuned against it.

## Context worth knowing

Not targets, and not things to watch on a dashboard — just how the pieces
behave, recorded so nobody has to rediscover them.

- The service runs on one Hetzner CX33 box. No second machine, no failover.
- Rate limiting is 1 request per second with a burst of 5. Timing ten rapid
  requests against production returned five results and rate-limited the rest,
  which is exactly the configured behaviour.
- Classification happens while mail is being delivered: the Cloudflare Email
  Worker waits for a verdict before forwarding. It gives up after 10 seconds and
  delivers the message anyway, so a slow classifier delays mail rather than
  losing it.
- The service keeps working when the LLM providers are unreachable. It records
  `llm_unavailable` and answers from the remaining stages.
- `/v1/classify` writes one structured log line per request with per-stage
  timings. Grafana Alloy ships these, and the runtime dashboard shows latency,
  error rate, degraded decisions, and which provider answered.

## Reproducing the measurements

Start the service locally, then send classifications with unique text and time
them. No repository code is involved in measuring — timing lives outside the
codebase, so nothing is shaped to fit a number.
