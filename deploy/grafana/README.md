# Sicurre ML observability

The production Alloy container and dashboard provisioner read Grafana Cloud
credentials from the server-owned `deploy/env.alloy`; copy
`deploy/env.alloy.example`, populate every value, and keep the resulting file
uncommitted with mode `0600`.

CD provisions `dashboards/sicurre-ml-runtime.json` only after `/v1/health` and
`/v1/ready` succeed. It runs the provisioner on the production server and reads
all Grafana values exclusively from `deploy/env.alloy`:

- `GRAFANA_URL`;
- `GRAFANA_SERVICE_ACCOUNT_TOKEN`, with folder,
  datasource, and dashboard read/write access.

The Alloy access-policy tokens only write telemetry and cannot replace the
Grafana service-account token used by the dashboard HTTP API. No Grafana token
needs to be duplicated in GitHub Actions secrets.

The runtime dashboard intentionally contains metrics only. Logs and traces are
explored through Grafana Logs Drilldown and Traces Drilldown.

The provisioner resolves the exact datasource names
`grafanacloud-sicurre-prom`, `grafanacloud-sicurre-logs`, and
`grafanacloud-sicurre-traces`, creates the `Sicurre ML` folder if absent, and
upserts dashboard UID `sicurre-ml-runtime`.
