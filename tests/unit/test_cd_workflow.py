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


def test_observability_validation_uses_running_app_network() -> None:
    workflow = Path(".github/workflows/cd.yml").read_text(encoding="utf-8")
    observability_job = workflow.split("observability-check:", maxsplit=1)[1].split(
        "provision-dashboard:", maxsplit=1
    )[0]

    assert "docker exec" in observability_job
    assert "OBSERVABILITY_APP_URL=http://127.0.0.1:8000" in observability_job
    assert "OBSERVABILITY_ALLOY_URL=http://alloy:12345" in observability_job
    assert "docker cp deploy/scripts/validate_observability.py" in observability_job
    assert "docker run" not in observability_job
    assert "--network" not in observability_job
    assert "/app/.venv/bin/python /tmp/validate_observability.py" in observability_job


def test_dashboard_provisioning_uses_container_python() -> None:
    workflow = Path(".github/workflows/cd.yml").read_text(encoding="utf-8")
    dashboard_job = workflow.split("provision-dashboard:", maxsplit=1)[1]

    assert "docker run --rm" in dashboard_job
    assert "/app/.venv/bin/python /workspace/provision_dashboard.py" in dashboard_job
    assert "\n              python /workspace/provision_dashboard.py" not in dashboard_job
