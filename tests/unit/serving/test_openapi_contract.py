"""Regression tests for the published inference API contract."""

from pathlib import Path

import yaml

from src.serving.app import app


def test_checked_in_openapi_matches_runtime() -> None:
    """Keep the reviewed API document synchronized with FastAPI."""
    published = yaml.safe_load(Path("docs/api/openapi.yaml").read_text(encoding="utf-8"))

    assert published == app.openapi()


def test_gateway_payload_bounds_are_published() -> None:
    """Expose the exact payload limits shared with the Sicurre gateway."""
    schema = app.openapi()["components"]["schemas"]["ClassifyRequest"]["properties"]

    assert schema["subject"]["maxLength"] == 500
    assert schema["text"]["maxLength"] == 5500
