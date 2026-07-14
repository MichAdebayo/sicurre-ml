from __future__ import annotations

import httpx

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


def test_resilient_post_retries_only_transient_statuses(monkeypatch) -> None:
    responses = [
        httpx.Response(503, request=httpx.Request("POST", "https://provider.test")),
        httpx.Response(200, request=httpx.Request("POST", "https://provider.test")),
    ]
    calls = 0

    def fake_post(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal calls
        response = responses[calls]
        calls += 1
        return response

    llm_classifier._circuit_failures.clear()
    llm_classifier._circuit_opened_at.clear()
    monkeypatch.setenv("LLM_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(llm_classifier.httpx, "post", fake_post)
    monkeypatch.setattr(llm_classifier.time, "sleep", lambda _: None)

    response = llm_classifier._resilient_post(
        "test-provider", "https://provider.test"
    )

    assert response.status_code == 200
    assert calls == 2


def test_resilient_post_does_not_retry_permanent_client_error(monkeypatch) -> None:
    calls = 0

    def fake_post(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal calls
        calls += 1
        return httpx.Response(
            400, request=httpx.Request("POST", "https://provider.test")
        )

    llm_classifier._circuit_failures.clear()
    llm_classifier._circuit_opened_at.clear()
    monkeypatch.setenv("LLM_MAX_ATTEMPTS", "3")
    monkeypatch.setattr(llm_classifier.httpx, "post", fake_post)

    response = llm_classifier._resilient_post(
        "test-provider", "https://provider.test"
    )

    assert response.status_code == 400
    assert calls == 1
