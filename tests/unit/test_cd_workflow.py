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
