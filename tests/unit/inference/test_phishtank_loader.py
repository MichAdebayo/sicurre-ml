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

def test_the_query_guards_the_json_cast() -> None:
    """data_raw_record is not all JSON, and one bad row failed the whole query.

    Mailbox exports and dropzone TXT files store raw message text. Postgres may
    evaluate ``raw_content::jsonb`` before the source filter, so without a guard
    a single non-JSON row raises and the blocklist silently drops to whatever
    the fallback file holds - one URL, against 1,153 in the database.
    """
    import inspect

    from src.inference import phishtank_loader

    source = inspect.getsource(phishtank_loader)
    assert "pg_input_is_valid(raw_content, 'jsonb')" in source


def test_a_failed_database_load_reports_its_cause(monkeypatch, capsys) -> None:
    """A stale credential and an unparseable row looked identical before."""
    from src.inference import phishtank_loader

    def _boom() -> list[str]:
        raise RuntimeError("password authentication failed")

    monkeypatch.setattr(phishtank_loader, "_load_phishtank_urls_from_database", _boom)
    monkeypatch.setattr(phishtank_loader, "_load_phishtank_urls_from_file", lambda: [])
    monkeypatch.setenv("PHISHTANK_SOURCE", "database")

    phishtank_loader.load_phishtank_urls()

    out = capsys.readouterr().out
    assert "RuntimeError" in out
    assert "password authentication failed" in out


def test_sqlite_urls_are_recognised_and_the_path_is_exact() -> None:
    """SQLAlchemy spells absolute paths with four slashes, relative with three.

    Keeping an extra slash makes SQLite read the first path segment as a URI
    authority and the load fails with "invalid uri authority".
    """
    from src.inference.phishtank_loader import _sqlite_path

    assert _sqlite_path("sqlite+aiosqlite:////Users/a/b.db") == "/Users/a/b.db"
    assert _sqlite_path("sqlite:///relative/c.db") == "relative/c.db"
    assert _sqlite_path("postgresql+psycopg://u@h/db") is None


def test_a_sqlite_data_platform_is_read_directly(tmp_path, monkeypatch) -> None:
    """The POC runs the data platform on SQLite.

    Without this the only way to get a blocklist locally was to point at a
    Postgres URL, so development reached for the production database.
    """
    import json
    import sqlite3

    db = tmp_path / "dp.db"
    conn = sqlite3.connect(db)
    conn.execute("create table data_raw_record (raw_content text)")
    conn.executemany(
        "insert into data_raw_record values (?)",
        [
            (json.dumps({"source": "phishtank_api", "url": "http://bad.test/a"}),),
            (json.dumps({"source": "phishtank_api", "url": "http://bad.test/b/"}),),
            (json.dumps({"source": "other", "url": "http://ignored.test"}),),
            ("Objet: not json at all",),  # mailbox export; must not break the query
        ],
    )
    conn.commit()
    conn.close()

    from src.inference import phishtank_loader

    monkeypatch.setenv("SICURRE_DATA_PLATFORM_DATABASE_URL", f"sqlite:///{db}")
    urls = phishtank_loader._load_phishtank_urls_from_database()

    assert urls == ["http://bad.test/a", "http://bad.test/b"]
