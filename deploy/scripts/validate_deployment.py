#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("INFERENCE_INTERNAL_URL", "http://app:8000").rstrip("/")
VALIDATION_HOST = os.getenv("INFERENCE_VALIDATION_HOST", "inference.sicurre.internal").strip()


def _decode_json_body(raw_body: bytes, *, status_code: int) -> dict[str, Any]:
    if not raw_body:
        return {}
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        if status_code >= 400:
            # TrustedHostMiddleware and reverse proxies can return plain text.
            # Keep diagnostics bounded and never echo an upstream body.
            return {"error": "non_json_http_error"}
        raise RuntimeError("Successful deployment response was not valid JSON") from None
    if not isinstance(body, dict):
        raise RuntimeError("Deployment response must be a JSON object")
    return body


def _request(
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    authenticated: bool = False,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    headers = {"Content-Type": "application/json"}
    if VALIDATION_HOST:
        headers["Host"] = VALIDATION_HOST
    if authenticated:
        headers["Authorization"] = f"Bearer {os.environ['INFERENCE_API_KEY']}"
    request = Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=20) as response:  # noqa: S310
            raw_body = response.read()
            return (
                response.status,
                _decode_json_body(raw_body, status_code=response.status),
                dict(response.headers.items()),
            )
    except HTTPError as exc:
        return (
            exc.code,
            _decode_json_body(exc.read(), status_code=exc.code),
            dict(exc.headers.items()),
        )


def _wait_for(path: str, expected_status: int, attempts: int) -> dict[str, Any]:
    for _ in range(attempts):
        status, body, _ = _request(path)
        if status == expected_status:
            return body
        time.sleep(5)
    raise RuntimeError(f"{path} did not return HTTP {expected_status}")


def main() -> None:
    _wait_for("/v1/health", 200, attempts=60)
    _wait_for("/v1/ready", 200, attempts=180)

    status, body, headers = _request(
        "/v1/classify",
        method="POST",
        authenticated=True,
        payload={
            "subject": "Deployment validation",
            "sender": "monitor@sicurre.internal",
            "text": "Message de validation interne sans lien.",
            "use_virustotal": False,
            "use_llm": False,
        },
    )
    if status != 200:
        raise RuntimeError(f"Authenticated inference smoke test returned HTTP {status}")

    required_fields = {
        "verdict",
        "label_verdict",
        "is_phishing",
        "composite_score",
        "stage_scores",
        "stage_labels",
        "label_distribution",
        "stage_breakdown",
        "explanation",
        "llm_provider",
    }
    if not required_fields.issubset(body):
        raise RuntimeError("Inference response contract is incomplete")
    if body["verdict"] not in {"safe", "phishing"}:
        raise RuntimeError("Inference verdict is outside the public contract")

    required_headers = {
        "X-Sicurre-Service-Version",
        "X-Sicurre-Model-Version",
        "X-Sicurre-Model-Revision",
        "X-Sicurre-Deployment-Revision",
    }
    normalized_headers = {name.lower(): value for name, value in headers.items()}
    if not {name.lower() for name in required_headers}.issubset(normalized_headers):
        raise RuntimeError("Inference identity headers are incomplete")

    manifest_status, manifest, _ = _request("/v1/manifest", authenticated=True)
    if manifest_status != 200 or manifest.get("schema_version") != 1:
        raise RuntimeError("Deployment manifest contract validation failed")

    output_path = os.getenv("DEPLOYMENT_MANIFEST_OUTPUT")
    if output_path:
        Path(output_path).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print("Deployment validation passed: health, readiness, auth, response, and identity.")


if __name__ == "__main__":
    main()
