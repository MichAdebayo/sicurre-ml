import json
import re
from pathlib import Path


def test_deployment_validation_uses_running_app_environment() -> None:
    workflow = Path(".github/workflows/cd.yml").read_text(encoding="utf-8")
    validation_job = workflow.split("health-check:", maxsplit=1)[1].split(
        "observability-check:", maxsplit=1
    )[0]

    assert "docker exec" in validation_job
    assert "INFERENCE_INTERNAL_URL=http://127.0.0.1:8000" in validation_job
    assert "INFERENCE_VALIDATION_HOST=inference.sicurre.internal" in validation_job
    assert "docker cp deploy/scripts/validate_deployment.py" in validation_job
    assert "--env-file .env" not in validation_job
    assert "deploy/current-deployment.json.tmp" in validation_job
    assert "/app/.venv/bin/python /tmp/validate_deployment.py" in validation_job
    assert "if ! docker exec" in validation_job
    assert "if ! docker cp" in validation_job
    assert '--env EXPECTED_MODEL_REVISION="$expected_model_revision"' in validation_job
    assert '--env EXPECTED_MODEL_VERSION="$expected_model_version"' in validation_job
    assert "Cannot validate an unpinned model revision." in validation_job


def test_observability_validation_uses_container_local_endpoints() -> None:
    workflow = Path(".github/workflows/cd.yml").read_text(encoding="utf-8")
    observability_job = workflow.split("observability-check:", maxsplit=1)[1].split(
        "provision-dashboard:", maxsplit=1
    )[0]

    assert "docker run --rm" in observability_job
    assert "OBSERVABILITY_PHASE=generate" in observability_job
    assert "OBSERVABILITY_APP_URL=http://127.0.0.1:8000" in observability_job
    assert "OBSERVABILITY_APP_HOST=inference.sicurre.internal" in observability_job
    assert "docker compose -f docker-compose.prod.yml ps -q alloy" in observability_job
    assert '--network "container:$alloy_container"' in observability_job
    assert "OBSERVABILITY_PHASE=delivery" in observability_job
    assert "OBSERVABILITY_ALLOY_URL=http://127.0.0.1:12345" in observability_job
    assert "docker compose -f docker-compose.prod.yml logs --tail=100 alloy" in observability_job
    assert "validate_observability.py:/tmp/validate_observability.py:ro" in observability_job
    assert "/app/.venv/bin/python /tmp/validate_observability.py" in observability_job


def test_cd_force_recreates_alloy_after_config_sync() -> None:
    workflow = Path(".github/workflows/cd.yml").read_text(encoding="utf-8")

    assert "docker compose -f docker-compose.prod.yml up -d --force-recreate alloy" in workflow
    assert "ML-owned Alloy failed to remain running" in workflow
    assert "docker compose -f docker-compose.prod.yml logs --tail=150 alloy" in workflow
    assert "https://*/loki/api/v1/push" in workflow
    assert "must be an HTTPS Loki push endpoint" in workflow
    assert "GRAFANA_PROMETHEUS_WRITE_API_TOKEN" in workflow
    assert "deploy/env.alloy is missing a non-empty ${required_var}" in workflow
    assert "HF_MODEL_REVISION must be an immutable 40-character lowercase HF commit" in workflow
    assert "MISTRAL_MODEL must use a pinned provider model ID" in workflow


def test_ci_starts_pinned_alloy_runtime_graph() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "alloy-runtime:" in workflow
    assert "alloy-runtime-check" in workflow
    assert "http://127.0.0.1:12345/-/ready" in workflow
    assert "grafana/alloy:v1.16.1@sha256:" in workflow


def test_dashboard_provisioning_uses_container_python() -> None:
    workflow = Path(".github/workflows/cd.yml").read_text(encoding="utf-8")
    dashboard_job = workflow.split("provision-dashboard:", maxsplit=1)[1]

    assert "docker run --rm" in dashboard_job
    assert "/app/.venv/bin/python /workspace/provision_dashboard.py" in dashboard_job
    assert "\n              python /workspace/provision_dashboard.py" not in dashboard_job


def test_remote_cd_scripts_do_not_require_host_python() -> None:
    workflow = Path(".github/workflows/cd.yml").read_text(encoding="utf-8")

    assert re.search(r"(?m)^\s+python(?:3)?\s", workflow) is None
    assert "previous_digest=$(sed -n 's/^CONTAINER_IMAGE_DIGEST=//p' .env | tail -1)" in workflow


def test_alloy_uses_shared_drilldown_service_identity() -> None:
    config = Path("deploy/alloy/config.alloy").read_text(encoding="utf-8")

    assert 'replacement   = "sicurre-ml-inference"' in config
    assert 'replacement   = "sicurre-ml-alloy"' in config
    assert '"service_name" = "sicurre-ml-inference"' in config
    assert '"service_name" = "sicurre-ml-alloy"' in config
    assert 'password = sys.env("GRAFANA_PROMETHEUS_WRITE_API_TOKEN")' in config
    assert "GRAFANA_PROMETHEUS_METRICS_API_TOKEN" not in config
    remote_write = config.split('prometheus.remote_write "grafana_cloud"', maxsplit=1)[1].split(
        "// Meta-monitor", maxsplit=1
    )[0]
    assert "service_name" not in remote_write
    assert 'encoding.from_json(sys.env("OTEL_TRACE_SAMPLE_PERCENT"))' in config
    assert "convert.to_number" not in config
    assert 'loki.source.api "sicurre_ml_smoke"' in config
    assert 'listen_address = "127.0.0.1"' in config
    assert 'key       = "http.status_code"' in config
    assert 'key       = "http.response.status_code"' in config
    assert '"__address__"  = "inference.sicurre.internal:8000"' in config


def test_production_app_emits_candidate_traces_to_ml_alloy() -> None:
    compose = Path("docker-compose.prod.yml").read_text(encoding="utf-8")

    app = compose.split("  app:", maxsplit=1)[1].split("  alloy:", maxsplit=1)[0]
    assert 'OTEL_EXPORTER_OTLP_ENDPOINT: "http://alloy:4317"' in app
    assert 'OTEL_EXPORTER_OTLP_INSECURE: "true"' in app
    assert 'OTEL_TRACE_SAMPLE_RATIO: "${OTEL_TRACE_SAMPLE_RATIO:-1.0}"' in app
    assert "- inference.sicurre.internal" in app
    assert "GRAFANA_" not in app


def test_dashboard_and_alerts_distinguish_app_from_alloy() -> None:
    dashboard = json.loads(
        Path("deploy/grafana/dashboards/sicurre-ml-runtime.json").read_text(encoding="utf-8")
    )
    alerts = json.loads(
        Path("deploy/grafana/alerts/sicurre-ml-alerts.json").read_text(encoding="utf-8")
    )

    panels = {p["title"]: p for p in dashboard["panels"] if p["type"] != "row"}
    for row in (p for p in dashboard["panels"] if p["type"] == "row"):
        panels.update({p["title"]: p for p in row.get("panels", [])})

    service_up = panels["ML scrape"]
    assert 'service_name="sicurre-ml-inference"' in service_up["targets"][0]["expr"]
    alert_expressions = {alert["uid"]: alert["expr"] for alert in alerts}
    assert 'service_name="sicurre-ml-inference"' in alert_expressions["sicurre-ml-unavailable"]
    assert 'service_name="sicurre-ml-alloy"' in alert_expressions["sicurre-ml-telemetry-scrape"]


def _ml_dashboard_panels() -> tuple[dict, dict]:
    """Flatten the dashboard, including panels nested inside collapsed rows."""
    dashboard = json.loads(
        Path("deploy/grafana/dashboards/sicurre-ml-runtime.json").read_text(encoding="utf-8")
    )
    panels = {p["title"]: p for p in dashboard["panels"] if p["type"] != "row"}
    for row in (p for p in dashboard["panels"] if p["type"] == "row"):
        panels.update({p["title"]: p for p in row.get("panels", [])})
    return dashboard, panels


def test_dashboard_opens_on_a_screenshottable_first_view() -> None:
    """The first screen must carry charts, not two rows of stat cards.

    The dashboard previously spent its first ten grid units on stat cards and did
    not reach a latency chart until y=21, so a 1440x900 capture showed no chart at
    all. The first view is now a compact summary strip over two chart rows, and
    everything else sits in collapsed sections.
    """
    dashboard, _ = _ml_dashboard_panels()

    rows = [p for p in dashboard["panels"] if p["type"] == "row" and p["collapsed"]]
    assert rows, "detail must be grouped into collapsed rows"
    assert all(r["collapsed"] for r in rows), "detail rows must start collapsed"

    first_view = min(r["gridPos"]["y"] for r in rows)
    assert first_view <= 18, (
        f"the first view is {first_view} grid units tall and will not fit a "
        f"1440x900 capture without scrolling"
    )

    charts_above_fold = [
        p
        for p in dashboard["panels"]
        if p["type"] in {"timeseries", "bargauge"} and p["gridPos"]["y"] < first_view
    ]
    assert len(charts_above_fold) >= 4, (
        "the first view must show latency, activity, classification and stage "
        "duration without scrolling"
    )
    assert {p["title"] for p in charts_above_fold} == {
        "Inference latency", "Request activity", "Content classification", "Mean stage duration"
    }

    for band_y in {p["gridPos"]["y"] for p in dashboard["panels"] if p["type"] != "row"}:
        width = sum(
            p["gridPos"]["w"]
            for p in dashboard["panels"]
            if p["type"] != "row" and p["gridPos"]["y"] == band_y
        )
        assert width == 24, f"row at y={band_y} occupies {width} of 24 columns"


def test_dashboard_separates_the_service_from_its_telemetry_agent() -> None:
    """Scrape health is the inference service, never the whole stack.

    stack="sicurre-ml" also matches the Alloy agent, so an unscoped `up` would
    report healthy whenever either process is scraped.
    """
    _, panels = _ml_dashboard_panels()

    expr = panels["ML scrape"]["targets"][0]["expr"]
    assert 'service_name="sicurre-ml-inference"' in expr


def test_dashboard_reports_identity_and_units_precisely() -> None:
    """Model revision is truncated for display; resource units are real units."""
    _, panels = _ml_dashboard_panels()

    assert panels["Model revision"]["options"]["textMode"] == "name"
    model_target = panels["Model revision"]["targets"][0]
    assert "display_version" in model_target["expr"]
    assert "$1…$2" in model_target["expr"]
    assert model_target["legendFormat"] == "{{display_version}}"

    assert panels["Process memory"]["fieldConfig"]["defaults"]["unit"] == "bytes"
    assert panels["Process CPU"]["fieldConfig"]["defaults"]["unit"] == "cores"

    assert (
        "increase(sicurre_inference_label_total"
        in panels["Content classification"]["targets"][0]["expr"]
    )


def test_dashboard_distinguishes_absent_telemetry_from_measured_zero() -> None:
    """Sparse event counters may read zero only with healthy request telemetry."""
    dashboard, panels = _ml_dashboard_panels()

    degraded = panels["Degraded decisions"]["targets"][0]["expr"]
    assert "sicurre_inference_degradation_total" in degraded, (
        "degradation is its own metric, distinct from error_total"
    )
    assert "vector(0)" not in degraded
    assert "0 * sum(increase(sicurre_inference_requests_total" in degraded
    assert 'service_name="sicurre-ml-inference"' in degraded
    assert "== 1" in degraded

    all_panels = [p for p in dashboard["panels"] if p["type"] != "row"]
    for row in (p for p in dashboard["panels"] if p["type"] == "row"):
        all_panels.extend(row.get("panels", []))
    for panel in all_panels:
        assert panel["fieldConfig"]["defaults"].get("noValue") in {"No data", "Unavailable"}, (
            f"{panel['title']} does not distinguish absent data from zero"
        )


def test_dashboard_does_not_imply_an_unreachable_latency_threshold() -> None:
    """No panel may cite a latency threshold the histogram cannot reach.

    The buckets and the alert threshold only mean something together, so this
    reads the ceiling from the emitter rather than repeating a literal. Re-cutting
    the buckets then updates the assertion instead of stranding it.
    """
    from src.serving.telemetry import _PROMETHEUS_BUCKETS_MS

    _, panels = _ml_dashboard_panels()
    ceiling = int(max(_PROMETHEUS_BUCKETS_MS))

    latency = panels["Inference latency"]
    assert str(ceiling) in latency["description"], (
        f"the latency panel must state the histogram ceiling ({ceiling} ms)"
    )


def test_dashboard_uses_emitted_provider_labels_and_observed_stage_samples() -> None:
    _, panels = _ml_dashboard_panels()
    provider = panels["LLM provider outcomes"]["targets"][0]
    assert "provider, category" in provider["expr"]
    assert "{{category}}" in provider["legendFormat"]
    stage = panels["Mean stage duration"]["targets"][0]["expr"]
    assert " > 0" in stage
    assert "clamp_min" not in stage


def test_observability_smoke_forces_privacy_safe_trace_and_auth_log() -> None:
    validator = Path("deploy/scripts/validate_observability.py").read_text(encoding="utf-8")

    assert '"Authorization": "Bearer observability-validation-invalid"' in validator
    assert '"traceparent":' in validator
    assert "Request(" in validator
    assert "telemetry_delivery_validation" in validator
    assert "loki_source_docker_target_entries_total" in validator
