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
    assert "docker cp deploy/scripts/validate_deployment.py" in validation_job
    assert "--env-file .env" not in validation_job
    assert "deploy/current-deployment.json.tmp" in validation_job
    assert "/app/.venv/bin/python /tmp/validate_deployment.py" in validation_job


def test_observability_validation_uses_container_local_endpoints() -> None:
    workflow = Path(".github/workflows/cd.yml").read_text(encoding="utf-8")
    observability_job = workflow.split("observability-check:", maxsplit=1)[1].split(
        "provision-dashboard:", maxsplit=1
    )[0]

    assert "docker run --rm" in observability_job
    assert "OBSERVABILITY_PHASE=generate" in observability_job
    assert "OBSERVABILITY_APP_URL=http://127.0.0.1:8000" in observability_job
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

    service_up = next(
        panel for panel in dashboard["panels"] if panel["title"] == "Metrics scrape health"
    )
    assert 'service_name="sicurre-ml-inference"' in service_up["targets"][0]["expr"]
    alert_expressions = {alert["uid"]: alert["expr"] for alert in alerts}
    assert 'service_name="sicurre-ml-inference"' in alert_expressions[
        "sicurre-ml-unavailable"
    ]
    assert 'service_name="sicurre-ml-alloy"' in alert_expressions[
        "sicurre-ml-telemetry-scrape"
    ]


def test_observability_smoke_forces_privacy_safe_trace_and_auth_log() -> None:
    validator = Path("deploy/scripts/validate_observability.py").read_text(encoding="utf-8")

    assert '"Authorization": "Bearer observability-validation-invalid"' in validator
    assert '"traceparent":' in validator
    assert "Request(" in validator
    assert "telemetry_delivery_validation" in validator
    assert "loki_source_docker_target_entries_total" in validator
