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


def test_model_text_matches_the_training_corpus_shape() -> None:
    """The classifier must be fed the framing it was fine-tuned on.

    The training corpus is "Objet : <subject>", a blank line, then the body, and
    it carries no sender field whatsoever. Labelling the parts at inference time
    ("Domaine expéditeur :", "Message :") put tokens in front of the model that
    it never saw while learning, which cost measurable phishing recall and made
    the verdict fragile to an appended sign-off.
    """
    result = canonicalize_email(
        "Merci de confirmer vos coordonnées bancaires.",
        subject="Régularisation de votre dossier",
        sender="service@urssaf-regul.top",
    )

    assert result.model_text == (
        "Objet : Régularisation de votre dossier\n\n"
        "Merci de confirmer vos coordonnées bancaires.\n\n"
        "urssaf-regul.top"
    )
    assert "Domaine expéditeur" not in result.model_text
    assert "Message :" not in result.model_text


def test_model_text_keeps_the_sender_domain_available_to_the_classifier() -> None:
    """The domain carries real signal - abused TLDs above all - so dropping the
    label must not drop the domain with it."""
    result = canonicalize_email(
        "Votre colis est bloqué.", subject="Suivi", sender="no-reply@suivi-colis.gq"
    )

    assert result.model_text.endswith("suivi-colis.gq")
    assert result.sender_domain == "suivi-colis.gq"


def test_model_text_omits_the_domain_when_no_sender_is_supplied() -> None:
    """A missing sender must not append the literal placeholder to the text."""
    result = canonicalize_email("Bonjour, voici le compte rendu.", subject="Réunion")

    assert result.model_text == "Objet : Réunion\n\nBonjour, voici le compte rendu."
    assert "non-fourni" not in result.model_text
