#!/usr/bin/env python3
"""Idempotently provision the metrics-only Sicurre ML Grafana dashboard."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

FOLDER_UID = "sicurre-ml"
FOLDER_TITLE = "Sicurre ML"
PROMETHEUS_DATASOURCE = "grafanacloud-sicurre-prom"


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _request(
    base_url: str,
    token: str,
    endpoint: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    accepted: tuple[int, ...] = (200,),
) -> tuple[int, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        f"{base_url.rstrip('/')}{endpoint}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310
            status = response.status
            body = json.loads(response.read() or b"{}")
    except HTTPError as exc:
        status = exc.code
        raw_body = exc.read()
        body = json.loads(raw_body or b"{}")
    if status not in accepted:
        message = body.get("message", "unknown Grafana API error")
        raise RuntimeError(f"{method} {endpoint} failed with HTTP {status}: {message}")
    return status, body


def _replace_strings(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_strings(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, old, new) for key, item in value.items()}
    return value


def _alert_rule(definition: dict[str, Any], datasource_uid: str) -> dict[str, Any]:
    evaluator_type = str(definition["operator"])
    return {
        "uid": definition["uid"],
        "folderUID": FOLDER_UID,
        "ruleGroup": "Sicurre ML production",
        "title": definition["title"],
        "condition": "C",
        "data": [
            {
                "refId": "A",
                "queryType": "",
                "relativeTimeRange": {"from": 900, "to": 0},
                "datasourceUid": datasource_uid,
                "model": {
                    "editorMode": "code",
                    "expr": definition["expr"],
                    "instant": True,
                    "intervalMs": 60000,
                    "legendFormat": "__auto",
                    "maxDataPoints": 43200,
                    "refId": "A",
                },
            },
            {
                "refId": "B",
                "queryType": "",
                "relativeTimeRange": {"from": 0, "to": 0},
                "datasourceUid": "-100",
                "model": {
                    "conditions": [],
                    "datasource": {"type": "__expr__", "uid": "-100"},
                    "expression": "A",
                    "reducer": "last",
                    "refId": "B",
                    "type": "reduce",
                },
            },
            {
                "refId": "C",
                "queryType": "",
                "relativeTimeRange": {"from": 0, "to": 0},
                "datasourceUid": "-100",
                "model": {
                    "conditions": [
                        {
                            "evaluator": {
                                "params": [definition["threshold"]],
                                "type": evaluator_type,
                            },
                            "operator": {"type": "and"},
                            "query": {"params": ["C"]},
                            "reducer": {"params": [], "type": "last"},
                            "type": "query",
                        }
                    ],
                    "datasource": {"type": "__expr__", "uid": "-100"},
                    "expression": "B",
                    "refId": "C",
                    "type": "threshold",
                },
            },
        ],
        "noDataState": definition.get("no_data", "OK"),
        "execErrState": "Error",
        "for": definition["for"],
        "annotations": {"summary": definition["title"]},
        "labels": {"service": "sicurre-ml", "severity": definition["severity"]},
        "isPaused": False,
    }


def _provision_alerts(
    base_url: str,
    token: str,
    datasource_uid: str,
    alert_path: Path,
) -> None:
    definitions = json.loads(alert_path.read_text(encoding="utf-8"))
    _, existing = _request(
        base_url,
        token,
        "/api/v1/provisioning/alert-rules",
    )
    existing_uids = {rule.get("uid") for rule in existing}
    for definition in definitions:
        rule = _alert_rule(definition, datasource_uid)
        if definition["uid"] in existing_uids:
            endpoint = f"/api/v1/provisioning/alert-rules/{definition['uid']}"
            method = "PUT"
        else:
            endpoint = "/api/v1/provisioning/alert-rules"
            method = "POST"
        _request(
            base_url,
            token,
            endpoint,
            method=method,
            payload=rule,
            accepted=(200, 201, 202),
        )


def main() -> None:
    dashboard_path = Path(
        sys.argv[1] if len(sys.argv) > 1 else "/workspace/dashboards/sicurre-ml-runtime.json"
    )
    alert_path = dashboard_path.parent.parent / "alerts" / "sicurre-ml-alerts.json"
    grafana_url = _required_env("GRAFANA_URL")
    token = _required_env("GRAFANA_SERVICE_ACCOUNT_TOKEN")

    _, datasource = _request(
        grafana_url,
        token,
        f"/api/datasources/name/{PROMETHEUS_DATASOURCE}",
    )
    datasource_uid = str(datasource["uid"])

    folder_status, _ = _request(
        grafana_url,
        token,
        f"/api/folders/{FOLDER_UID}",
        accepted=(200, 404),
    )
    if folder_status == 404:
        _request(
            grafana_url,
            token,
            "/api/folders",
            method="POST",
            payload={"uid": FOLDER_UID, "title": FOLDER_TITLE},
            accepted=(200, 201, 409, 412),
        )

    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    dashboard = _replace_strings(dashboard, "__PROM_UID__", datasource_uid)
    dashboard["id"] = None
    _, provisioned = _request(
        grafana_url,
        token,
        "/api/dashboards/db",
        method="POST",
        payload={
            "dashboard": dashboard,
            "folderUid": FOLDER_UID,
            "overwrite": True,
            "message": "Provisioned by sicurre-ml CD",
        },
        accepted=(200, 201),
    )

    _, verified = _request(
        grafana_url,
        token,
        f"/api/dashboards/uid/{dashboard['uid']}",
    )
    if verified.get("dashboard", {}).get("uid") != dashboard["uid"]:
        raise RuntimeError("Grafana dashboard verification failed")
    _provision_alerts(grafana_url, token, datasource_uid, alert_path)
    print(f"Provisioned {dashboard['title']}: {grafana_url.rstrip('/')}{provisioned['url']}")


if __name__ == "__main__":
    main()
