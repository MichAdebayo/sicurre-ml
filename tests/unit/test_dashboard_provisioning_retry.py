"""Bounded retry for Grafana dashboard provisioning.

CD run 31937993160 built the image, deployed to Hetzner, validated the
deployment and validated telemetry delivery, then failed provisioning on
`503 {"code":"Loading"}` from a suspended free-tier Grafana instance. The
release was good; only the dashboard step was not retried.
"""

from __future__ import annotations

import importlib.util
import sys
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "deploy" / "grafana" / "provision_dashboard.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("provision_dashboard", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["provision_dashboard"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


provision = _load_module()


class _Response:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _loading_error() -> HTTPError:
    return HTTPError(
        url="https://grafana.test/api/datasources/name/grafanacloud-sicurre-prom",
        code=503,
        msg="Service Unavailable",
        hdrs=Message(),
        fp=None,
    )


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provision.time, "sleep", lambda _: None)


def test_backoff_grows_but_stays_capped() -> None:
    assert provision._retry_delay_seconds(1) == 2
    assert provision._retry_delay_seconds(2) == 4
    assert provision._retry_delay_seconds(3) == 8
    assert provision._retry_delay_seconds(50) == 30


def test_recovers_once_the_instance_finishes_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact failure from CD run 31937993160."""
    attempts: list[int] = []

    def fake_urlopen(request, timeout=30):  # noqa: ARG001
        attempts.append(1)
        if len(attempts) <= 2:
            error = _loading_error()
            error.read = lambda: (  # type: ignore[method-assign]
                b'{"code":"Loading","message":"Your instance is loading, '
                b'and will be ready shortly."}'
            )
            raise error
        return _Response(b'{"uid":"prom-uid"}')

    monkeypatch.setattr(provision, "urlopen", fake_urlopen)

    status, body = provision._request(
        "https://grafana.test", "token", "/api/datasources/name/x"
    )

    assert status == 200
    assert body == {"uid": "prom-uid"}
    assert len(attempts) == 3


def test_auth_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeating a 401 only delays the real error."""
    attempts: list[int] = []

    def fake_urlopen(request, timeout=30):  # noqa: ARG001
        attempts.append(1)
        error = HTTPError(url="x", code=401, msg="Unauthorized", hdrs=Message(), fp=None)
        error.read = lambda: b'{"message":"invalid API key"}'  # type: ignore[method-assign]
        raise error

    monkeypatch.setattr(provision, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="invalid API key"):
        provision._request("https://grafana.test", "token", "/api/datasources/name/x")

    assert len(attempts) == 1


def test_network_faults_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def fake_urlopen(request, timeout=30):  # noqa: ARG001
        attempts.append(1)
        if len(attempts) == 1:
            raise URLError("connection reset")
        return _Response(b"{}")

    monkeypatch.setattr(provision, "urlopen", fake_urlopen)

    status, _ = provision._request("https://grafana.test", "token", "/x")

    assert status == 200
    assert len(attempts) == 2


def test_retry_budget_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A permanently unavailable Grafana must fail the deploy, not hang it."""
    attempts: list[int] = []

    def fake_urlopen(request, timeout=30):  # noqa: ARG001
        attempts.append(1)
        error = _loading_error()
        error.read = lambda: b'{"message":"still loading"}'  # type: ignore[method-assign]
        raise error

    monkeypatch.setattr(provision, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="after 6 attempts"):
        provision._request("https://grafana.test", "token", "/x")

    assert len(attempts) == provision.MAX_ATTEMPTS


def test_non_json_error_body_does_not_mask_the_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gateway HTML page must not surface as a JSON parse error."""
    assert provision._decode(b"<html>502 Bad Gateway</html>") == {
        "message": "<html>502 Bad Gateway</html>"
    }
    assert provision._decode(b"") == {}


def test_list_bodies_keep_their_shape() -> None:
    """`/api/datasources` answers with an array.

    Coercing it into a dict made every caller that iterates the response walk
    dict keys instead, which is what broke CD run 31942362873 with
    "'str' object has no attribute 'get'".
    """
    assert provision._decode(b'[{"uid":"a"},{"uid":"b"}]') == [
        {"uid": "a"},
        {"uid": "b"},
    ]
    assert provision._error_message([{"uid": "a"}]) == "unknown Grafana API error"
    assert provision._error_message({"message": "boom"}) == "boom"


@pytest.mark.parametrize(
    "written",
    ['glsa_token', '"glsa_token"', "'glsa_token'", ' glsa_token ', '"glsa_token" '],
)
def test_token_is_normalised_however_the_env_file_was_written(
    monkeypatch: pytest.MonkeyPatch, written: str
) -> None:
    """`docker run --env-file` does not strip quotes or whitespace.

    An identical token authenticated in the repository that shell-sources the
    same file and returned HTTP 401 in the one that passes it to Docker, which
    is what failed CD run 31942804573.
    """
    monkeypatch.setenv("GRAFANA_SERVICE_ACCOUNT_TOKEN", written)

    assert provision._required_env("GRAFANA_SERVICE_ACCOUNT_TOKEN") == "glsa_token"


def test_missing_token_still_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAFANA_SERVICE_ACCOUNT_TOKEN", '""')

    with pytest.raises(RuntimeError, match="is required"):
        provision._required_env("GRAFANA_SERVICE_ACCOUNT_TOKEN")


def test_preflight_reports_a_rejected_credential_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surface the cause in seconds instead of after a full deploy."""

    def fake_urlopen(request, timeout=30):  # noqa: ARG001
        error = HTTPError(url="x", code=401, msg="Unauthorized", hdrs=Message(), fp=None)
        error.read = lambda: b'{"message":"Unauthorized"}'  # type: ignore[method-assign]
        raise error

    monkeypatch.setattr(provision, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="stray quotes or whitespace"):
        provision._verify_credentials("https://grafana.test", "bad-token")


def test_preflight_passes_a_working_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        provision, "urlopen", lambda *_, **__: _Response(b'{"id":1,"name":"Main Org."}')
    )

    provision._verify_credentials("https://grafana.test", "glsa_good")


def test_preflight_does_not_mask_an_unavailable_grafana(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A waking instance is a retry case, not a credential problem."""
    attempts: list[int] = []

    def fake_urlopen(request, timeout=30):  # noqa: ARG001
        attempts.append(1)
        error = _loading_error()
        error.read = lambda: b'{"message":"still loading"}'  # type: ignore[method-assign]
        raise error

    monkeypatch.setattr(provision, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="after 6 attempts"):
        provision._verify_credentials("https://grafana.test", "glsa_good")
    assert len(attempts) == provision.MAX_ATTEMPTS
