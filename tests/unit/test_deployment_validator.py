import runpy
from pathlib import Path
from typing import Any, Callable, cast
from urllib.error import URLError

import pytest


def _decoder() -> Callable[..., dict[str, Any]]:
    namespace = runpy.run_path(str(Path("deploy/scripts/validate_deployment.py")))
    return cast(Callable[..., dict[str, Any]], namespace["_decode_json_body"])


def test_http_error_body_may_be_plain_text() -> None:
    decode = _decoder()

    assert decode(b"Invalid host header", status_code=400) == {
        "error": "non_json_http_error"
    }


def test_valid_json_object_is_preserved() -> None:
    decode = _decoder()

    assert decode(b'{"status":"ok"}', status_code=200) == {"status": "ok"}


def test_wait_for_retries_transient_startup_connection_failure() -> None:
    namespace = runpy.run_path(str(Path("deploy/scripts/validate_deployment.py")))
    wait_for = cast(Callable[..., dict[str, Any]], namespace["_wait_for"])
    attempts = 0

    def request(*_args: Any, **_kwargs: Any) -> tuple[int, dict[str, Any], dict[str, str]]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise URLError("connection refused")
        return 200, {"status": "ok"}, {}

    wait_for.__globals__["_request"] = request
    wait_for.__globals__["time"].sleep = lambda _seconds: None

    assert wait_for("/v1/health", 200, attempts=2) == {"status": "ok"}
    assert attempts == 2


def test_validator_requires_exact_promoted_model_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path(str(Path("deploy/scripts/validate_deployment.py")))
    main = cast(Callable[[], None], namespace["main"])
    main.__globals__["_wait_for"] = lambda *_args, **_kwargs: {}
    response = {
        "verdict": "safe",
        "label_verdict": "legitimate",
        "is_phishing": False,
        "composite_score": 0.1,
        "stage_scores": {},
        "stage_labels": {},
        "label_distribution": {},
        "stage_breakdown": {},
        "explanation": "bounded",
        "llm_provider": None,
    }
    headers = {
        "X-Sicurre-Service-Version": "0.1.0",
        "X-Sicurre-Model-Version": "1.0.17",
        "X-Sicurre-Model-Revision": "a" * 40,
        "X-Sicurre-Deployment-Revision": "image-sha",
    }
    model_identity: dict[str, Any] = {
        "version": "1.0.17",
        "requested_revision": "a" * 40,
        "revision": "a" * 40,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model": model_identity,
    }

    def request(path: str, **_kwargs: Any) -> tuple[int, dict[str, Any], dict[str, str]]:
        return (200, manifest, {}) if path == "/v1/manifest" else (200, response, headers)

    main.__globals__["_request"] = request
    monkeypatch.setenv("EXPECTED_MODEL_REVISION", "a" * 40)
    monkeypatch.setenv("EXPECTED_MODEL_VERSION", "1.0.17")
    main()

    model_identity["revision"] = "b" * 40
    with pytest.raises(RuntimeError, match="immutable model revision"):
        main()
