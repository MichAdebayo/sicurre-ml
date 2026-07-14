"""Load the PhishTank blocklist from Neon, with a warmed file fallback."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path


def _normalize_db_conninfo(conninfo: str) -> str:
    if conninfo.startswith("postgresql+psycopg://"):
        return conninfo.replace("postgresql+psycopg://", "postgresql://", 1)
    if conninfo.startswith("postgresql+psycopg2://"):
        return conninfo.replace("postgresql+psycopg2://", "postgresql://", 1)
    return conninfo


def _canonicalize_urls(urls: list[str]) -> list[str]:
    canonical = {
        url.strip().rstrip("/")
        for url in urls
        if isinstance(url, str) and url.strip()
    }
    return sorted(canonical)


def _fallback_path() -> Path | None:
    path = os.getenv("PHISHTANK_FILE_PATH")
    return Path(path) if path else None


def _load_phishtank_urls_from_database() -> list[str]:
    conninfo = os.getenv("SICURRE_DATA_PLATFORM_DATABASE_URL")
    if not conninfo:
        raise RuntimeError("SICURRE_DATA_PLATFORM_DATABASE_URL is not configured")

    import psycopg

    query = """
    select distinct raw_content::jsonb->>'url' as url
    from public.data_raw_record
    where raw_content::jsonb->>'source' = 'phishtank_api'
      and coalesce(raw_content::jsonb->>'url', '') <> ''
    order by url;
    """

    with psycopg.connect(_normalize_db_conninfo(conninfo)) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    return _canonicalize_urls([row[0] for row in rows])


def _write_fallback_file(urls: list[str]) -> None:
    path = _fallback_path()
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "neon_db",
        "synced_at": datetime.now(UTC).isoformat(),
        "count": len(urls),
        "urls": urls,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _load_phishtank_urls_from_file() -> list[str]:
    path = _fallback_path()
    if path is None:
        print("[phishtank] PHISHTANK_FILE_PATH not configured; continuing without PhishTank URLs.")
        return []

    if not path.exists():
        print("[phishtank] File fallback unavailable; continuing without PhishTank URLs.")
        return []

    with path.open() as fh:
        urls = json.load(fh)["urls"]
    urls = _canonicalize_urls(urls)
    print(f"[phishtank] Loaded {len(urls)} URLs from file.")
    return urls


def load_phishtank_urls() -> list[str]:
    """Return a list of known phishing URLs from the configured source.

    Environment variables
    ---------------------
    PHISHTANK_SOURCE          : "database" (default) or "file"
                                Legacy value "http" is treated as "database".
    SICURRE_DATA_PLATFORM_DATABASE_URL
                              : SQLAlchemy-style or PostgreSQL URL for Neon
    PHISHTANK_FILE_PATH       : path to a local phishtank_urls.json fallback
    """
    source = os.getenv("PHISHTANK_SOURCE", "database").lower()
    if source == "http":
        source = "database"

    if source != "file":
        try:
            urls = _canonicalize_urls(_load_phishtank_urls_from_database())
            print(f"[phishtank] Loaded {len(urls)} URLs from Neon database.")
            try:
                _write_fallback_file(urls)
            except Exception:
                print("[phishtank] Fallback write skipped (write_failed).")
            return urls
        except Exception:
            print("[phishtank] Database load failed (source_unavailable); using fallback.")

    return _load_phishtank_urls_from_file()


@lru_cache(maxsize=1)
def get_phishtank_set() -> frozenset[str]:
    """Cached frozenset of known phishing URLs for O(1) lookup."""
    return frozenset(load_phishtank_urls())
