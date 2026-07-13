#!/usr/bin/env bash
set -euo pipefail

: "${GRAFANA_URL:?GRAFANA_URL is required}"
dashboard_token="${GRAFANA_SERVICE_ACCOUNT_TOKEN:-${GRAFANA_API_TOKEN:-}}"
: "${dashboard_token:?GRAFANA_SERVICE_ACCOUNT_TOKEN is required}"

dashboard_path="${1:-deploy/grafana/dashboards/sicurre-ml-runtime.json}"
auth_header="Authorization: Bearer ${dashboard_token}"

datasource_uid() {
  curl --fail --silent --show-error \
    -H "$auth_header" \
    "${GRAFANA_URL%/}/api/datasources/name/$1" | jq -er '.uid'
}

prom_uid=$(datasource_uid "grafanacloud-sicurre-prom")

folder_uid=$(
  curl --fail --silent --show-error \
    -H "$auth_header" \
    "${GRAFANA_URL%/}/api/folders/sicurre-ml" | jq -r '.uid // empty' || true
)
if [ -z "$folder_uid" ]; then
  folder_uid=$(
    curl --fail --silent --show-error -X POST \
      -H "$auth_header" -H "Content-Type: application/json" \
      --data '{"uid":"sicurre-ml","title":"Sicurre ML"}' \
      "${GRAFANA_URL%/}/api/folders" | jq -er '.uid'
  )
fi

jq \
  --arg prom "$prom_uid" \
  --arg folder "$folder_uid" \
  '{dashboard: (. | walk(if type == "string" then
      gsub("__PROM_UID__"; $prom)
    else . end)), folderUid: $folder, overwrite: true, message: "Provisioned by sicurre-ml CD"}' \
  "$dashboard_path" | \
  curl --fail --silent --show-error -X POST \
    -H "$auth_header" -H "Content-Type: application/json" \
    --data-binary @- "${GRAFANA_URL%/}/api/dashboards/db"
