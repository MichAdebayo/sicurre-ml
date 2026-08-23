from __future__ import annotations

from types import SimpleNamespace

import httpx

from src.inference import blocklist


def test_no_urls_is_clean_without_loading_phishtank(monkeypatch) -> None:
    monkeypatch.setattr(
        blocklist,
        "get_phishtank_set",
        lambda: (_ for _ in ()).throw(AssertionError("must not load")),
    )

    result = blocklist.check_blocklists("Message without a link")

    assert result.source == "clean"
    assert result.detail == "No URLs"


def test_whitelisted_subdomain_is_not_flagged(monkeypatch) -> None:
    monkeypatch.setattr(blocklist, "get_phishtank_set", lambda: set())

    result = blocklist.check_blocklists("https://account.microsoft.com/security")

    assert result.is_known_phishing is False


def test_phishtank_exact_match_is_flagged(monkeypatch) -> None:
    monkeypatch.setattr(
        blocklist,
        "get_phishtank_set",
        lambda: {"https://malicious.invalid/login"},
    )

    result = blocklist.check_blocklists("Open https://malicious.invalid/login/")

    assert result.source == "phishtank"
    assert result.confidence == 0.99


def test_french_dark_domain_is_flagged(monkeypatch) -> None:
    monkeypatch.setattr(blocklist, "get_phishtank_set", lambda: set())

    result = blocklist.check_blocklists("https://ameli-remboursement.invalid/login")

    assert result.source == "dark_list"
    assert result.confidence == 0.95


def test_virustotal_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("VIRUSTOTAL_API_KEY", raising=False)

    assert blocklist._query_virustotal("https://example.invalid") is None


def test_virustotal_malicious_result(monkeypatch) -> None:
    monkeypatch.setenv("VIRUSTOTAL_API_KEY", "test-token")
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://virustotal.test"),
        json={
            "data": {
                "attributes": {
                    "last_analysis_stats": {"malicious": 4, "harmless": 6}
                }
            }
        },
    )
    monkeypatch.setattr(blocklist.httpx, "get", lambda *args, **kwargs: response)

    result = blocklist._query_virustotal("https://malicious.invalid")

    assert result is not None
    assert result.source == "virustotal"
    assert result.confidence == 0.9


def test_virustotal_404_submits_without_blocking(monkeypatch) -> None:
    monkeypatch.setenv("VIRUSTOTAL_API_KEY", "test-token")
    response = httpx.Response(
        404,
        request=httpx.Request("GET", "https://virustotal.test"),
    )
    submissions: list[dict[str, object]] = []
    monkeypatch.setattr(blocklist.httpx, "get", lambda *args, **kwargs: response)

    def fake_post(*args: object, **kwargs: object) -> SimpleNamespace:
        submissions.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        blocklist.httpx,
        "post",
        fake_post,
    )

    assert blocklist._query_virustotal("https://unknown.invalid") is None
    assert submissions[0]["data"] == {"url": "https://unknown.invalid"}


def test_virustotal_provider_failure_is_fail_open(monkeypatch, capsys) -> None:
    monkeypatch.setenv("VIRUSTOTAL_API_KEY", "test-token")
    monkeypatch.setattr(
        blocklist.httpx,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.TimeoutException("slow")),
    )

    assert blocklist._query_virustotal("https://unknown.invalid") is None
    assert "provider_unavailable" in capsys.readouterr().out


def test_check_blocklists_uses_optional_virustotal(monkeypatch) -> None:
    monkeypatch.setattr(blocklist, "get_phishtank_set", lambda: set())
    expected = blocklist.BlocklistResult(True, 0.8, "virustotal")
    monkeypatch.setattr(blocklist, "_query_virustotal", lambda _: expected)

    assert blocklist.check_blocklists(
        "https://unlisted.invalid/path", use_virustotal=True
    ) is expected
