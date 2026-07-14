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

    assert config.count('service_name = "sicurre-ml-inference"') == 2
    assert config.count('stack        = "sicurre-ml"') == 2


def test_observability_smoke_forces_privacy_safe_trace_and_auth_log() -> None:
    validator = Path("deploy/scripts/validate_observability.py").read_text(encoding="utf-8")

    assert '"Authorization": "Bearer observability-validation-invalid"' in validator
    assert '"traceparent":' in validator
    assert "Request(" in validator
    assert "data=" not in validator
