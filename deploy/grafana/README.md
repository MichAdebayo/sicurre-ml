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
