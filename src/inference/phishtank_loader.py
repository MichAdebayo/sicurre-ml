"""Load the PhishTank blocklist — HTTP (preferred) or local file fallback."""

from __future__ import annotations

import json
import os
from functools import lru_cache

import httpx


def load_phishtank_urls() -> list[str]:
    """Return a list of known phishing URLs from the configured source.

    Environment variables
    ---------------------
    PHISHTANK_SOURCE          : "http" or "file" (default: "http")
    PHISHTANK_ENDPOINT_URL    : full URL to the sicurre app snapshot endpoint
    INTERNAL_API_KEY           : bearer token for the internal endpoint
    PHISHTANK_FILE_PATH       : path to a local phishtank_urls.json fallback
    """
    source = os.getenv("PHISHTANK_SOURCE", "http")

    if source == "http":
        endpoint = os.environ["PHISHTANK_ENDPOINT_URL"]
        key = os.environ["INTERNAL_API_KEY"]
        try:
            resp = httpx.get(
                endpoint,
                headers={"Authorization": f"Bearer {key}"},
                timeout=30,
            )
            resp.raise_for_status()
            urls: list[str] = resp.json()["urls"]
            print(f"[phishtank] Loaded {len(urls)} URLs via HTTP endpoint.")
            return urls
        except Exception as exc:
            print(f"[phishtank] HTTP load failed ({exc}), falling back to file.")

    # file fallback
    path = os.environ["PHISHTANK_FILE_PATH"]
    with open(path) as fh:
        urls = json.load(fh)["urls"]
    print(f"[phishtank] Loaded {len(urls)} URLs from file.")
    return urls


@lru_cache(maxsize=1)
def get_phishtank_set() -> frozenset[str]:
    """Cached frozenset of known phishing URLs for O(1) lookup."""
    return frozenset(load_phishtank_urls())
