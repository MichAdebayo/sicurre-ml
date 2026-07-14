# Sicurre ML CX33 performance baseline

Run only against an explicitly approved environment. The default script target
is production and requires the internal service credential.

```bash
K6_CLOUD_TOKEN=... \
INFERENCE_API_KEY=... \
PROFILE=warm USE_LLM=false \
k6 cloud run tests/e2e/k6_sicurre_ml.js
```

Profiles: `cold`, `warm`, `concurrent5`, `sustained` (1 request/second for 15
minutes), and `burst5` (5 requests/second for 30 seconds). Repeat relevant
profiles with `USE_LLM=true`. The maximum planned run is far below the assigned
200 k6 VUh ceiling; measured VUh must still be recorded after execution.

Capture p50/p95/p99 latency, throughput, error and 429 rates, CPU, resident
memory, recovery behavior, and Grafana ingestion usage. Stop the test if model
readiness fails, server errors exceed 2%, or memory continues rising without
stabilizing. Production execution remains unverified until an operator runs and
records the approved matrix.
