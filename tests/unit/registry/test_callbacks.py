from __future__ import annotations

import io
from email.message import Message
from urllib.error import HTTPError

import pytest

from src.registry.callbacks import post_provenance_callback


class _Response:
    headers = {"Content-Type": "application/json"}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"id":"record-1","status":"candidate","idempotent":false}'


def test_callback_returns_bounded_response_without_logging_payload() -> None:
    result = post_provenance_callback(
        base_url="https://sicurre.com",
        path="/internal/ml/candidates",
        bearer_token="secret",
        payload={"mlflow_run_id": "run-1"},
        opener=lambda *args, **kwargs: _Response(),
    )

    assert result.id == "record-1"
    assert result.idempotent is False


def test_callback_retries_503_then_succeeds() -> None:
    attempts = 0
    delays: list[float] = []

    def opener(*args: object, **kwargs: object) -> _Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            headers = Message()
            raise HTTPError("url", 503, "unavailable", headers, io.BytesIO())
        return _Response()

    post_provenance_callback(
        base_url="https://sicurre.com",
        path="/internal/ml/evaluations",
        bearer_token="secret",
        payload={"mlflow_evaluation_run_id": "eval-1"},
        opener=opener,
        sleeper=delays.append,
        jitter=lambda: 0.0,
    )

    assert attempts == 2
    assert delays == [1.0]


def test_callback_does_not_retry_422() -> None:
    headers = Message()

    def opener(*args: object, **kwargs: object) -> _Response:
        raise HTTPError("url", 422, "invalid", headers, io.BytesIO())

    with pytest.raises(RuntimeError, match="HTTP 422"):
        post_provenance_callback(
            base_url="https://sicurre.com",
            path="/internal/ml/evaluations",
            bearer_token="secret",
            payload={},
            opener=opener,
            sleeper=lambda _: None,
        )
