"""Stage 1 — Rule-based URL heuristics, French-adapted.

Checks URL structure, TLDs, homograph attacks, shorteners, and French brand
impersonation. Returns a RuleResult without any external I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

try:
    import tldextract as _tldextract  # optional at import time
except ModuleNotFoundError:  # pragma: no cover
    _tldextract = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# French & international brand homograph targets
# ---------------------------------------------------------------------------
BRAND_VARIATIONS: dict[str, list[str]] = {
    # French government / public services
    "ameli": ["ameli-fr", "ameli-securite", "ameli-remboursement", "ameli-compte"],
    "impots": ["impots-gouv", "impots-service", "imp0ts", "impotsgouvfr"],
    "caf": ["caf-fr", "caf-aide", "caf-alloc", "caf-service"],
    "cpam": ["cpam-fr", "cpam-securite", "cpam-sante"],
    "urssaf": ["urssaf-fr", "urssaf-service", "urssaf-compte"],
    "laposte": ["laposte-fr", "la-poste-fr", "lapost3", "laposte-colis"],
    "sncf": ["sncf-fr", "sncf-billet", "sncf-compte"],
    "edf": ["edf-fr", "edf-energie", "edf-client"],
    "engie": ["engie-fr", "engie-client", "eng1e"],
    "franceconnect": ["franceconnect-fr", "france-connect"],
    "fransetravail": ["france-travail-fr", "francetravail-fr"],
    # French banks
    "creditagricole": ["credit-agricole-fr", "ca-securite", "creditagr1cole"],
    "bnpparibas": ["bnp-paribas-fr", "bnppar1bas", "bnp-securite"],
    "societegenerale": ["societegenerale-fr", "socgen-securite", "sg-banque"],
    "caissedepargne": ["caissedepargne-fr", "caisse-depargne-securite"],
    "lcl": ["lcl-fr", "lcl-banque-fr", "lcl-securite"],
    # Telecoms / utilities used in France
    "orange": ["orange-fr", "orange-service", "0range", "orange-compte"],
    "sfr": ["sfr-fr", "sfr-service", "sfr-compte"],
    "bouygues": ["bouygues-fr", "bouyguestelecom-fr"],
    "free": ["free-fr", "free-service", "free-mobile-fr"],
    # International brands (widely phished in France)
    "paypal": ["paypa1", "paypai", "paypal-securite", "paypal-verify", "paypal-fr"],
    "microsoft": ["micr0soft", "microsft", "microsoft-fr", "microsoft-securite"],
    "google": ["g00gle", "googIe", "google-fr"],
    "amazon": ["amaz0n", "amazom", "amazon-fr", "amazon-livraison"],
    "apple": ["app1e", "appl3", "apple-fr"],
    "netflix": ["netf1ix", "netflix-fr"],
    "chronopost": ["chronopost-fr", "chron0post", "chronopost-colis"],
    "colissimo": ["colissimo-fr", "col1ssimo"],
}

# French phishing keywords likely to appear in malicious domains
FRENCH_PHISHING_KEYWORDS = [
    "connexion", "compte", "virement", "remboursement", "colis", "livraison",
    "secure", "securite", "verify", "verifier", "validation", "valider",
    "update", "mise-a-jour", "alert", "alerte", "suspend", "suspendu",
    "confirm", "confirmer", "login", "authentification", "acces", "access",
    "service", "client", "espace", "mon-espace", "mon-compte",
    "aide", "support", "assistance",
]

# Suspicious TLDs (free / abused registrars)
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work",
    "click", "pw", "cc", "biz", "info", "icu", "surf",
    "online", "site", "website", "space", "fun",
}

# Known URL shorteners
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "cutt.ly", "goo.gl", "t.co", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "rebrand.ly", "shorturl.at",
}

# Trusted French public-service domains (whitelist anchors)
FRENCH_WHITELIST_SUFFIXES = {
    "gouv.fr", "pole-emploi.fr", "france-travail.fr",
    "ameli.fr", "caf.fr", "cpam.fr", "urssaf.fr",
    "laposte.fr", "sncf.com", "edf.fr", "engie.fr",
    "creditagricole.fr", "bnpparibas.fr", "societegenerale.fr",
    "caissedepargne.fr", "lcl.fr",
    "orange.fr", "sfr.fr", "bouyguestelecom.fr", "free.fr",
    "impots.gouv.fr",
}


@dataclass
class RuleResult:
    is_phishing: bool
    confidence: float  # 0.0–1.0
    reasons: list[str] = field(default_factory=list)
    risk_score: int = 0


def _extract(url: str) -> tuple[str, str, str] | None:
    """Return (subdomain, domain, suffix) or None if tldextract unavailable."""
    if _tldextract is None:
        return None
    ext = _tldextract.extract(url)
    return ext.subdomain, ext.domain, ext.suffix


def check_url_rules(text: str) -> RuleResult:
    """Run all URL heuristic rules against *text* and return a RuleResult."""
    url_re = re.compile(
        r"https?://[^\s<>\"'{}|\\^`\[\]]+"
    )
    urls = url_re.findall(text)

    if not urls:
        return RuleResult(is_phishing=False, confidence=0.0, reasons=["No URLs found"])

    reasons: list[str] = []
    risk_score = 0
    max_single = 0

    def add(score: int, reason: str) -> None:
        nonlocal risk_score, max_single
        risk_score += score
        max_single = max(max_single, score)
        reasons.append(reason)

    for url in urls:
        try:
            parsed = urlparse(url)
            host = parsed.netloc.split(":")[0].lower()

            extracted = _extract(url)
            if extracted is None:
                # fallback: just use the raw host
                subdomain, domain_part, suffix = "", host, ""
            else:
                subdomain, domain_part, suffix = extracted
                domain_part = domain_part.lower()
                suffix = suffix.lower()
                full_domain = f"{subdomain}.{domain_part}.{suffix}".lstrip(".")

                # Whitelist: skip further checks for trusted FR public domains
                for trusted in FRENCH_WHITELIST_SUFFIXES:
                    if full_domain == trusted or full_domain.endswith("." + trusted):
                        continue

                # Rule 1: IP address URL (CRITICAL)
                if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host):
                    add(60, f"IP address URL: {url}")

                # Rule 2: Suspicious TLD
                if suffix in SUSPICIOUS_TLDS:
                    add(35, f"Suspicious TLD: .{suffix}")

                # Rule 3: URL shortener
                if any(s in f"{domain_part}.{suffix}" for s in URL_SHORTENERS):
                    add(40, f"URL shortener: {domain_part}.{suffix}")

                # Rule 4: Excessive subdomains
                sub_count = len([s for s in subdomain.split(".") if s])
                if sub_count > 2:
                    add(25, f"Excessive subdomains ({sub_count}): {full_domain}")

                # Rule 5: Homograph / brand impersonation
                for brand, variants in BRAND_VARIATIONS.items():
                    # Known typo variants
                    for v in variants:
                        if v in domain_part:
                            add(55, f"Typosquatting: '{domain_part}' impersonates '{brand}'")
                            break
                    else:
                        # Brand name in domain but not exact match + has digits
                        if (
                            brand in domain_part
                            and domain_part != brand
                            and re.search(r"\d", domain_part)
                        ):
                            add(
                                55,
                                f"Brand impersonation (digit substitution): "
                                f"'{domain_part}' looks like '{brand}'",
                            )

                # Rule 6: French phishing keywords in domain
                matched_kw = [kw for kw in FRENCH_PHISHING_KEYWORDS if kw in domain_part]
                if matched_kw:
                    kw_score = min(20 * len(matched_kw), 60)
                    add(kw_score, f"Phishing keywords in domain: {', '.join(matched_kw)}")

                # Rule 7: Suspicious TLD + brand keyword combo
                if suffix in SUSPICIOUS_TLDS:
                    brand_hits = [b for b in BRAND_VARIATIONS if b in domain_part]
                    if brand_hits:
                        add(30, f"Brand keyword + suspicious TLD: {brand_hits[0]}.{suffix}")

                # Rule 8: Dash/hyphen abuse
                if domain_part.count("-") >= 2:
                    add(25, f"Multiple hyphens in domain: {domain_part}")

                # Rule 9: Very long URL
                if len(url) > 150:
                    add(20, f"Excessively long URL ({len(url)} chars)")

                # Rule 10: @ symbol in URL (obfuscation)
                if "@" in url:
                    add(70, f"@ symbol in URL (obfuscation): {url}")

        except Exception:
            continue

    final_risk = max(risk_score, max_single)
    is_phishing = final_risk >= 30
    confidence = min(final_risk / 100.0, 0.99) if is_phishing else final_risk / 100.0

    return RuleResult(
        is_phishing=is_phishing,
        confidence=confidence,
        reasons=reasons,
        risk_score=final_risk,
    )
