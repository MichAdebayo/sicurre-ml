from __future__ import annotations

from src.inference import llm_classifier
from src.inference.llm_classifier import LLMResult


def test_user_prompt_includes_sender_subject_and_text() -> None:
    prompt = llm_classifier._user_prompt(
        text="Merci de confirmer votre compte.",
        sender="support@paypa1-security.com",
        subject="Action immediate requise",
    )

    assert "Expéditeur: support@paypa1-security.com" in prompt
    assert "Objet: Action immediate requise" in prompt
    assert "Merci de confirmer votre compte." in prompt


def test_classify_llm_forwards_sender_and_subject(monkeypatch) -> None:
    captured: dict[str, str | None] = {}

    def fake_tier(
        text: str,
        sender: str | None = None,
        subject: str | None = None,
    ) -> LLMResult:
        captured["text"] = text
        captured["sender"] = sender
        captured["subject"] = subject
        return LLMResult(
            label="phishing",
            confidence=0.85,
            explanation="Suspicious sender domain.",
            provider="fake",
        )

    monkeypatch.setattr(llm_classifier, "_TIERS", [fake_tier])

    result = llm_classifier.classify_llm(
        text="Veuillez reinitialiser votre mot de passe.",
        sender="alerts@micr0soft-security.com",
        subject="Votre compte sera suspendu",
    )

    assert result is not None
    assert result.provider == "fake"
    assert captured["text"] == "Veuillez reinitialiser votre mot de passe."
    assert captured["sender"] == "alerts@micr0soft-security.com"
    assert captured["subject"] == "Votre compte sera suspendu"
