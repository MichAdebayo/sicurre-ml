"""Stage 2 — Blocklist lookup: PhishTank + French dark list + VirusTotal (optional).

All lookups are URL/domain-level. No text classification here.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from src.inference.phishtank_loader import get_phishtank_set  # noqa: E402

# ---------------------------------------------------------------------------
# French whitelist — domains that should never be flagged
# ---------------------------------------------------------------------------
FRENCH_WHITELIST: frozenset[str] = frozenset(
    {
        # Government & public services
        "gouv.fr", "service-public.fr", "france.fr", "elysee.fr",
        "assemblee-nationale.fr", "senat.fr",
        "impots.gouv.fr", "dgfip.finances.gouv.fr",
        "ameli.fr", "caf.fr", "cpam.fr", "urssaf.fr",
        "pole-emploi.fr", "france-travail.fr",
        "laposte.fr", "laposte.net",
        "sncf.com", "sncf-connect.com",
        "edf.fr", "engie.fr",
        "mairie-de-paris.fr",
        # Banks
        "creditagricole.fr", "ca-paris.fr",
        "bnpparibas.fr", "mabanque.bnpparibas.fr",
        "societegenerale.fr",
        "caissedepargne.fr",
        "lcl.fr",
        "credit-mutuel.fr",
        "labanquepostale.fr",
        # Telecoms
        "orange.fr", "sosh.fr",
        "sfr.fr", "sfr.net",
        "bouyguestelecom.fr",
        "free.fr", "freemobile.fr",
        # Media & general
        "lemonde.fr", "lefigaro.fr", "liberation.fr",
        "france.tv", "france24.com", "rfi.fr",
        # International (widely used in France)
        "paypal.com", "microsoft.com", "apple.com",
        "google.com", "google.fr", "amazon.fr", "amazon.com",
        "netflix.com",
    }
)

# ---------------------------------------------------------------------------
# French dark list — hand-curated high-confidence phishing patterns
# These supplement PhishTank with FR-specific indicators.
# ---------------------------------------------------------------------------
FRENCH_DARK_DOMAINS: frozenset[str] = frozenset(
    {
        # Generic FR government impersonation patterns (substrings)
        "ameli-remboursement",
        "impots-service",
        "caf-aide-urgente",
        "cpam-remboursement",
        "urssaf-regularisation",
        "france-connect-validation",
        "laposte-colis-bloque",
        "sncf-remboursement-billet",
        # Generic phishing delivery patterns in FR
        "colis-en-attente",
        "livraison-impossible",
        "votre-colis-bloque",
        "remboursement-impots",
        "validation-compte-bancaire",
        "mise-a-jour-coordonnees",
        "suspension-compte",
    }
)


@dataclass
class BlocklistResult:
    is_known_phishing: bool
    confidence: float
    source: str  # "phishtank" | "dark_list" | "virustotal" | "clean"
    detail: str = ""


def _extract_domain(url: str) -> str:
    """Return the registered domain from a URL, e.g. 'paypal.com'."""
    try:
        import tldextract
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
    except Exception:
        pass
    # fallback
    parsed = urlparse(url)
    return parsed.netloc.lower().lstrip("www.")


def _extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s<>\"'{}|\\^`\[\]]+", text)


def _is_whitelisted(domain: str) -> bool:
    for trusted in FRENCH_WHITELIST:
        if domain == trusted or domain.endswith("." + trusted):
            return True
    return False


def check_blocklists(text: str, use_virustotal: bool = False) -> BlocklistResult:
    """Check all URLs in *text* against PhishTank, the French dark list,
    and (optionally) VirusTotal.

    Parameters
    ----------
    text:
        Raw SMS/email body.
    use_virustotal:
        When True and VIRUSTOTAL_API_KEY is set, look up each URL.
        Adds latency (~1–2 s per URL) so disabled by default in the hot path.
    """
    urls = _extract_urls(text)
    if not urls:
        return BlocklistResult(
            is_known_phishing=False, confidence=0.0, source="clean", detail="No URLs"
        )

    phishtank = get_phishtank_set()

    for url in urls:
        url_stripped = url.rstrip("/")
        domain = _extract_domain(url)

        if _is_whitelisted(domain):
            continue

        # PhishTank exact match
        if url_stripped in phishtank or url in phishtank:
            return BlocklistResult(
                is_known_phishing=True,
                confidence=0.99,
                source="phishtank",
                detail=f"Exact match in PhishTank: {url_stripped}",
            )

        # French dark list — substring match on domain
        for pattern in FRENCH_DARK_DOMAINS:
            if pattern in domain:
                return BlocklistResult(
                    is_known_phishing=True,
                    confidence=0.95,
                    source="dark_list",
                    detail=f"French dark list pattern '{pattern}' in domain '{domain}'",
                )

        # VirusTotal (optional, async-friendly via sync httpx for now)
        if use_virustotal:
            vt_result = _query_virustotal(url_stripped)
            if vt_result is not None:
                return vt_result

    return BlocklistResult(is_known_phishing=False, confidence=0.0, source="clean")


def _query_virustotal(url: str) -> BlocklistResult | None:
    """Query VirusTotal for a single URL. Returns None on API error.

    Free tier: 4 req/min, 500/day.
    Requires VIRUSTOTAL_API_KEY in environment.
    """
    api_key = os.getenv("VIRUSTOTAL_API_KEY")
    if not api_key:
        return None

    import base64

    headers = {"x-apikey": api_key}
    url_id = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")

    try:
        resp = httpx.get(
            f"https://www.virustotal.com/api/v3/urls/{url_id}",
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 404:
            # URL not in VT database yet — submit it asynchronously, don't block
            httpx.post(
                "https://www.virustotal.com/api/v3/urls",
                headers=headers,
                data={"url": url},
                timeout=5,
            )
            return None
        resp.raise_for_status()
        stats = resp.json()["data"]["attributes"]["last_analysis_stats"]
        malicious = stats.get("malicious", 0)
        total = sum(stats.values()) or 1
        if malicious >= 3:
            confidence = min(malicious / total + 0.5, 0.99)
            return BlocklistResult(
                is_known_phishing=True,
                confidence=confidence,
                source="virustotal",
                detail=f"{malicious}/{total} VT engines flagged as malicious",
            )
    except Exception as exc:
        print(f"[blocklist] VirusTotal error: {exc}")

    return None
