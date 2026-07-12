# Sicurre ML observability

The production Alloy container reads Grafana Cloud destination credentials from
the server-owned `deploy/env.alloy`; copy `deploy/env.alloy.example`, populate
all nine values, and keep the resulting file uncommitted with mode `0600`.

CD provisions `dashboards/sicurre-ml-runtime.json` only after `/v1/health` and
`/v1/ready` succeed. Add these repository secrets before the first deployment:

- `GRAFANA_URL`: the Grafana Cloud stack URL;
- `GRAFANA_API_TOKEN`: a service-account token with folder and dashboard
  read/write access.

The provisioner resolves the exact datasource names
`grafanacloud-sicurre-prom`, `grafanacloud-sicurre-logs`, and
`grafanacloud-sicurre-traces`, creates the `Sicurre ML` folder if absent, and
upserts dashboard UID `sicurre-ml-runtime`.
