#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

APP_URL = os.getenv("OBSERVABILITY_APP_URL", "http://127.0.0.1:8000").rstrip("/")
ALLOY_URL = os.getenv("OBSERVABILITY_ALLOY_URL", "http://alloy:12345").rstrip("/")
PHASE = os.getenv("OBSERVABILITY_PHASE", "all").strip().lower()


def _read(url: str) -> str:
    with urlopen(url, timeout=15) as response:  # noqa: S310
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        return response.read().decode()


def _read_with_retry(url: str, *, attempts: int = 12, delay_seconds: int = 5) -> str:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return _read(url)
        except (OSError, RuntimeError, URLError) as exc:
            last_error = exc
            time.sleep(delay_seconds)
    raise RuntimeError(f"{url} unavailable after {attempts} attempts") from last_error


def _metric_sum(metrics: str, metric_name: str) -> float:
    pattern = re.compile(rf"^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$")
    return sum(
        float(match.group(1))
        for line in metrics.splitlines()
        if (match := pattern.match(line))
    )


def _require_delivery(metrics: str, candidates: tuple[str, ...], pipeline: str) -> None:
    if not any(_metric_sum(metrics, name) > 0 for name in candidates):
        counters = ", ".join(f"{name}={_metric_sum(metrics, name):g}" for name in candidates)
        raise RuntimeError(f"{pipeline} has not recorded successful delivery ({counters})")


def _push_alloy_loki_smoke() -> None:
    request = Request(
        "http://127.0.0.1:3500/loki/api/v1/raw",
        data=b'{"event":"telemetry_delivery_validation","service":"sicurre-ml-inference"}\n',
        method="POST",
        headers={"Content-Type": "application/x-ndjson"},
    )
    with urlopen(request, timeout=15) as response:  # noqa: S310
        if response.status not in {200, 204}:
            raise RuntimeError(f"Alloy Loki smoke receiver returned HTTP {response.status}")


def _generate_app_telemetry() -> None:
    app_metrics = _read_with_retry(f"{APP_URL}/v1/metrics")
    if "sicurre_service_up 1" not in app_metrics:
        raise RuntimeError("Application telemetry endpoint is unhealthy")

    # Generate an authentication failure so the error tail-sampling policy and
    # security log path have deterministic post-deployment traffic.
    # A sampled W3C parent plus an invalid bearer deterministically exercises
    # the privacy-safe authentication log and error-trace paths. No email or
    # request body is sent.
    request = Request(
        f"{APP_URL}/v1/classify",
        method="POST",
        headers={
            "Authorization": "Bearer observability-validation-invalid",
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        },
    )
    try:
        urlopen(request, timeout=15)  # noqa: S310
    except HTTPError:
        pass
    print("Application telemetry generation passed: metrics, auth log, and sampled trace emitted.")


def _validate_alloy_delivery() -> None:
    _read_with_retry(f"{ALLOY_URL}/-/ready")
    _push_alloy_loki_smoke()

    # Allow at least one 60-second application and Alloy self-scrape cycle.
    time.sleep(65)
    alloy_metrics = _read_with_retry(f"{ALLOY_URL}/metrics")
    required_families = (
        "prometheus_remote_storage_",
        "loki_write_",
        "otelcol_",
    )
    missing = [family for family in required_families if family not in alloy_metrics]
    if missing:
        raise RuntimeError(f"Alloy telemetry families missing: {', '.join(missing)}")
    _require_delivery(
        alloy_metrics,
        ("loki_source_docker_target_entries_total",),
        "Docker log source",
    )
    _require_delivery(
        alloy_metrics,
        ("prometheus_remote_storage_samples_total",),
        "Prometheus remote write",
    )
    _require_delivery(
        alloy_metrics,
        ("loki_write_sent_entries_total", "loki_write_sent_bytes_total"),
        "Loki",
    )
    _require_delivery(
        alloy_metrics,
        ("otelcol_exporter_sent_spans", "otelcol_exporter_sent_spans_total"),
        "OTLP traces",
    )
    print("Observability validation passed: Alloy, scrape, Loki, and OTLP metrics present.")


def main() -> None:
    if PHASE not in {"all", "generate", "delivery"}:
        raise RuntimeError(f"Unsupported observability validation phase: {PHASE}")
    if PHASE in {"all", "generate"}:
        _generate_app_telemetry()
    if PHASE in {"all", "delivery"}:
        _validate_alloy_delivery()


if __name__ == "__main__":
    main()
