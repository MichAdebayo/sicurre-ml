"""Verify local-only docs against isolated instances of the real ML app."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ("development", 200),
        ("dev", 200),
        ("local", 200),
        ("production", 404),
        (" Production ", 404),
        ("prod", 404),
        ("staging", 404),
    ],
)
def test_documentation_visibility(environment: str, expected: int) -> None:
    """Import each environment independently without model prewarming or inference."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
from fastapi.testclient import TestClient
from src.serving.app import app
client = TestClient(app)
for path in ('/docs', '/redoc', '/openapi.json', '/docs/oauth2-redirect'):
    response = client.get(path)
    assert response.status_code == {expected}, (path, response.status_code)
assert client.get('/v1/health').status_code == 200
assert '/v1/classify' in app.openapi()['paths']
""",
        ],
        env={
            **os.environ,
            "DEPLOYMENT_ENV": environment,
            "OTEL_SDK_DISABLED": "true",
            "INFERENCE_ALLOWED_HOSTS": "testserver",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_production_compose_pins_environment() -> None:
    """A missing or stale server .env must not re-enable public documentation."""
    compose = yaml.safe_load(Path("docker-compose.prod.yml").read_text())
    assert compose["services"]["app"]["environment"]["DEPLOYMENT_ENV"] == "production"
