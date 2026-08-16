#!/usr/bin/env python3
"""Idempotently provision the metrics-only Sicurre ML Grafana dashboard."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

FOLDER_UID = "sicurre-ml"
FOLDER_TITLE = "Sicurre ML"
PROMETHEUS_DATASOURCE = "grafanacloud-sicurre-prom"

# Grafana Cloud suspends idle free-tier instances and answers
# `503 {"code":"Loading"}` while one wakes. CD run 31937993160 deployed, health
# checked, and validated telemetry successfully, then failed here on exactly
# that response.
#
# Only conditions that can clear on their own are retried. Auth, payload and
# not-found errors fail immediately, because repeating them only delays the
# real error. The retry is inlined rather than imported: this script is mounted
# into a container as a single file, so a sibling module is a bundling risk.
RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
MAX_ATTEMPTS = int(os.getenv("GRAFANA_PROVISION_MAX_ATTEMPTS", "6"))
RETRY_BASE_SECONDS = float(os.getenv("GRAFANA_PROVISION_RETRY_BASE_SECONDS", "2"))
RETRY_MAX_SECONDS = float(os.getenv("GRAFANA_PROVISION_RETRY_MAX_SECONDS", "30"))


def _retry_delay_seconds(attempt: int) -> float:
    """Exponential backoff capped so an outage cannot become an unbounded wait."""
    return min(RETRY_BASE_SECONDS * 2 ** (max(attempt, 1) - 1), RETRY_MAX_SECONDS)


def _required_env(name: str) -> str:
    """Read a required value, tolerating how the env file was written.

    CD passes `deploy/env.alloy` to `docker run --env-file`, which does not
    strip surrounding quotes or trailing whitespace the way shell sourcing
    does. A token written as `TOKEN="glsa_x"` therefore reached Grafana with
    literal quotes attached and produced HTTP 401, while the identical value
    authenticated fine in the sibling repository that sources the same file
    through a shell.
    """
    value = os.getenv(name, "").strip().strip("\"'").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _verify_credentials(grafana_url: str, token: str) -> None:
    """Prove the token authenticates before doing any provisioning work.

    The CD precheck only asserts that the variable is non-empty, so a
    malformed token passes validation and fails a few hundred lines later,
    after a full build and deploy. A side-effect-free call surfaces it in
    seconds and distinguishes a bad credential from an unavailable Grafana.
    """
    try:
        _request(grafana_url, token, "/api/org")
    except RuntimeError as exc:
        if "HTTP 401" in str(exc) or "HTTP 403" in str(exc):
            raise RuntimeError(
                "GRAFANA_SERVICE_ACCOUNT_TOKEN was rejected by Grafana "
                f"({exc}). Check deploy/env.alloy for stray quotes or "
                "whitespace around the value, then confirm the service "
                "account is still enabled."
            ) from exc
        raise


def _decode(raw: bytes) -> Any:
    """Parse a Grafana body, tolerating the HTML a gateway may return.

    The shape is preserved: several endpoints answer with a JSON array, and
    coercing those into a dict silently breaks every caller that iterates them.
    """
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        # A waking instance or proxy can answer with HTML. Keep the text so the
        # surfaced error describes the real failure, not a JSON parse error.
        return {"message": raw.decode("utf-8", "replace")[:200]}


def _error_message(body: Any) -> str:
    """Extract an error message from a body of any shape."""
    if isinstance(body, dict):
        return str(body.get("message", "unknown Grafana API error"))
    return "unknown Grafana API error"


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

    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = Request(
            f"{base_url.rstrip('/')}{endpoint}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        transient_reason: str | None = None
        try:
            with urlopen(request, timeout=30) as response:  # noqa: S310
                status = response.status
                body = _decode(response.read())
        except HTTPError as exc:
            status = exc.code
            body = _decode(exc.read())
            if status in RETRYABLE_STATUSES:
                transient_reason = f"HTTP {status}: {_error_message(body)}"
        except URLError as exc:
            # DNS, connection reset and timeouts are worth repeating.
            status = 0
            body = {}
            transient_reason = f"connection failed: {exc.reason}"

        if transient_reason is None:
            if status not in accepted:
                raise RuntimeError(
                    f"{method} {endpoint} failed with HTTP {status}: "
                    f"{_error_message(body)}"
                )
            return status, body

        if attempt == MAX_ATTEMPTS:
            raise RuntimeError(
                f"{method} {endpoint} failed after {MAX_ATTEMPTS} attempts: "
                f"{transient_reason}"
            )

        delay = _retry_delay_seconds(attempt)
        print(
            f"Grafana not ready (attempt {attempt}/{MAX_ATTEMPTS}), "
            f"retrying in {delay:.0f}s: {transient_reason}",
            flush=True,
        )
        time.sleep(delay)

    raise RuntimeError(f"{method} {endpoint} exhausted its retry budget")


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

    _verify_credentials(grafana_url, token)

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
