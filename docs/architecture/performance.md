# Performance and quality

## The bar

**Two seconds, end to end.** That is the number Sicurre is trying to beat for a
classification, because mail should not feel held up.

Everything below is either a measurement or context that explains one. None of
it is a dashboard requirement.

## What was measured

Measured 2 September 2026, against the production model (v15, Hugging Face
revision `86e90dc5`), classifying French emails without the LLM stage.

| Path | Median | p95 | Notes |
|------|--------|-----|-------|
| `rules` stage alone | 0.001 ms | 0.01 ms | 600 calls, in-process |
| `blocklist` stage alone (local list) | 0.001 ms | 0.01 ms | 600 calls, in-process |
| Full classify, same machine | 3 ms | 36 ms | 15 unique inputs, local server |
| Full classify, over the internet | 426 ms | 438 ms | 10 requests to `api.sicurre.com` |

The model itself costs a few milliseconds. Almost all of the 426 ms is network
and TLS between a laptop and the Hetzner box — from inside the same network it
would be far lower.

**Comfortably inside the two-second bar**, with room to spare.

Method: unique text per request so nothing could be served from cache, and the
loaded model confirmed via `/v1/manifest` before timing.

## What does not fit in two seconds

The LLM stage. It is bounded at 7.5 seconds total
(`LLM_TOTAL_TIMEOUT_SECONDS`, 2.5 s per provider) across a chain of Mistral then
Groq. When it is switched on, a single classification can take several seconds —
well past the bar.

This is worth knowing rather than acting on. The ONNX path is what meets the
target; the LLM is the slower, more accurate second opinion. If a request needs
to be fast, it runs with `use_llm=false`.

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
