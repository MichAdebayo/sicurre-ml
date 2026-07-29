from __future__ import annotations

from src.inference.input_normalizer import (
    canonicalize_email,
    minimize_for_external_llm,
)


def test_canonicalize_multipart_prefers_plain_text() -> None:
    raw = """MIME-Version: 1.0
Content-Type: multipart/alternative; boundary=mail

--mail
Content-Type: text/plain; charset=utf-8

Bonjour, votre inscription est confirmée.
--mail
Content-Type: text/html; charset=utf-8

<p>Contenu de secours</p>
--mail--
"""

    result = canonicalize_email(
        raw,
        subject="Confirmation",
        sender="personne@example.fr",
    )

    assert result.source_format == "mime"
    assert result.body == "Bonjour, votre inscription est confirmée."
    assert result.sender_domain == "example.fr"
    assert "Content-Type" not in result.model_text


def test_canonicalize_html_preserves_visible_text_and_link() -> None:
    result = canonicalize_email(
        '<p>Consultez votre facture.</p><a href="https://example.fr/facture">Ouvrir</a>'
    )

    assert result.source_format == "html"
    assert "Consultez votre facture." in result.body
    assert "https://example.fr/facture" in result.security_text


def test_external_projection_removes_direct_identifiers() -> None:
    projected = minimize_for_external_llm(
        "Écrire à jean.dupont@example.fr, appeler +33 6 12 34 56 78 "
        "et ouvrir https://example.fr/compte/jean?token=secret."
    )

    assert "jean.dupont" not in projected
    assert "+33" not in projected
    assert "token=secret" not in projected
    assert "[EMAIL domaine=example.fr]" in projected
    assert "[URL domaine=example.fr]" in projected
