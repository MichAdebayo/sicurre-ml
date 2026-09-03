# Sicurre ML observability

The production Alloy container and dashboard provisioner read Grafana Cloud
credentials from the server-owned `deploy/env.alloy`; copy
`deploy/env.alloy.example`, populate every value, and keep the resulting file
uncommitted with mode `0600`.

CD provisions `dashboards/sicurre-ml-runtime.json` only after `/v1/health` and
`/v1/ready` succeed. It runs the Python standard-library provisioner inside the
exact deployed Sicurre ML application image and passes all Grafana values
exclusively through `deploy/env.alloy`:

- `GRAFANA_URL`;
- `GRAFANA_SERVICE_ACCOUNT_TOKEN`, with folder,
  datasource, and dashboard read/write access.

The Alloy access-policy tokens only write telemetry and cannot replace the
Grafana service-account token used by the dashboard HTTP API. No Grafana token
needs to be duplicated in GitHub Actions secrets.

The runtime dashboard intentionally contains metrics only. Logs and traces are
explored through Grafana Logs Drilldown and Traces Drilldown.

No Python, Node, `jq`, or other parser is installed on or required from the
host. The repository-owned application image is the deterministic provisioning
runtime.

The provisioner resolves the canonical metrics datasource
`grafanacloud-sicurre-prom`, creates the `Sicurre ML` folder if absent, and
upserts dashboard UID `sicurre-ml-runtime`. Logs and traces use the shared
Grafana Cloud Loki and Tempo datasources; Drilldown separates this workload
with `stack="sicurre-ml"` and `service_name="sicurre-ml-inference"` (or
`sicurre-ml-alloy` for collector self-telemetry). Separate per-service
datasources are neither required nor provisioned.

## Dashboard semantics

The first view presents four roomy period-inference cards above four charts.
Service health, resources and reliability have separate expandable sections
below it. Health includes public HTTP, metrics scraping, model readiness, Alloy,
sample age and model revision; resources includes the active-series budget.
Current health is never inferred from historical traffic.
P50/P95 cards aggregate histogram observations across the selected period and
all requested modes, not averages of plotted quantiles. They measure ML handler
duration, not email delivery, and remain neutral because mode SLOs differ.
Missing observations remain missing; sparse percentiles are estimates. Counts
use Prometheus `increase`, so displayed whole numbers remain estimates over the
selected range, not the database's audit counts. An unobserved stage is omitted;
an event counter falls back to zero only with a healthy inference scrape and
an observed request counter. Provider outcomes are grouped by `provider` and
`category`, matching the emitted metric labels.

Health observations older than 180 seconds are stale. Once samples leave the
Prometheus lookback window they become unknown, never healthy by default.
Memory and active-series colors use their existing strict alert thresholds;
CPU has no invented threshold. CPU/memory describe the ML process, not the host.

## Public HTTP health probe

Alloy probes `https://api.sicurre.com/v1/health` every 60 seconds with a five-second
HTTP timeout. It requires status 200 and a JSON `status: ok` response; redirects,
bad response bodies and HTTP failures do not pass. TLS verification stays enabled.
The Compose environment provides `ML_HEALTH_PROBE_URL`, overridable in the server
root `.env`. Existing installations need no additional secret. `/v1/metrics`
remains the internal Prometheus scrape; `/v1/health` is not scraped as metrics.

Only five probe series are retained. The HTTP alert waits two minutes and treats
missing observations as alerting. This probe covers the public DNS/TLS/proxy route
from the ML host, not an authenticated inference request or an independent
external host monitor. Alloy failure also interrupts probe reporting.

The isolated runtime regression uses the exact repository blackbox module, a
local HTTP fixture and a local Prometheus receiver, with no production exports:

```bash
ALLOY_BINARY=/path/to/alloy PROMETHEUS_BINARY=/path/to/prometheus \
  uv run pytest tests/integration/test_monitoring_probe.py -q
```

The fixture checks success, malformed bodies, HTTP 503, redirects, connection
failure, recovery, stale samples and missing samples. The CI Alloy runtime job
also checks the complete configuration with a loopback-only probe target.

## Shared notifications

ML rules carry `service=sicurre-ml`. The shared Grafana notification policy is
owned by Sicurre's `scripts/deploy/notification-policy.mjs` and routes this label
as well as `stack=sicurre` to `Sicurre Operations`. Provisioning ML rules alone
does not create this route. The contact point sends to `michael@sicurre.com` and
includes resolved notifications.

The linked alert-chain dashboard belongs to Sicurre. Its bounded synthetic
exercise verifies signal ingestion, rule evaluation and email delivery without
interrupting either service. It is not a test of every ML detector or a real
customer outage. Keep the local incident reproduction and versioned correction
as separate C21 evidence.
