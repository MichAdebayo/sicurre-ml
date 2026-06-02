from __future__ import annotations

import httpx

from src.inference.phishtank_loader import get_phishtank_set, load_phishtank_urls


def test_load_phishtank_urls_prefers_database_and_writes_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    fallback = tmp_path / "phishtank_urls.json"
    monkeypatch.setenv("PHISHTANK_SOURCE", "database")
    monkeypatch.setenv("SICURRE_DATA_PLATFORM_DATABASE_URL", "postgresql+psycopg://example")
    monkeypatch.setenv("PHISHTANK_FILE_PATH", str(fallback))
    monkeypatch.setattr(
        "src.inference.phishtank_loader._load_phishtank_urls_from_database",
        lambda: ["https://example.test/a", "https://example.test/a/", "https://example.test/b"],
    )

    urls = load_phishtank_urls()

    assert urls == ["https://example.test/a", "https://example.test/b"]
    assert fallback.exists() is True
    assert '"count": 2' in fallback.read_text()


def test_load_phishtank_urls_treats_legacy_http_source_as_database(monkeypatch) -> None:
    monkeypatch.setenv("PHISHTANK_SOURCE", "http")
    monkeypatch.setattr(
        "src.inference.phishtank_loader._load_phishtank_urls_from_database",
        lambda: ["https://example.test/a"],
    )

    assert load_phishtank_urls() == ["https://example.test/a"]


def test_load_phishtank_urls_keeps_database_result_when_fallback_write_fails(monkeypatch) -> None:
    monkeypatch.setenv("PHISHTANK_SOURCE", "database")
    monkeypatch.setattr(
        "src.inference.phishtank_loader._load_phishtank_urls_from_database",
        lambda: ["https://example.test/a"],
    )

    def fail_write(urls) -> None:
        raise PermissionError("read only")

    monkeypatch.setattr(
        "src.inference.phishtank_loader._write_fallback_file",
        fail_write,
    )

    assert load_phishtank_urls() == ["https://example.test/a"]


def test_get_phishtank_set_returns_empty_when_http_and_file_fallback_fail(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PHISHTANK_SOURCE", "database")
    monkeypatch.setenv("SICURRE_DATA_PLATFORM_DATABASE_URL", "postgresql+psycopg://example")
    monkeypatch.setenv("PHISHTANK_FILE_PATH", "/tmp/does-not-exist.json")

    def fake_load(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        "src.inference.phishtank_loader._load_phishtank_urls_from_database",
        fake_load,
    )
    get_phishtank_set.cache_clear()

    assert get_phishtank_set() == frozenset()

    get_phishtank_set.cache_clear()