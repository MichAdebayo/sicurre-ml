from __future__ import annotations

import json

import httpx
import pytest

from src.inference import llm_classifier
from src.inference.llm_classifier import LLMResult
from src.inference.mail_context import MailContext


def test_user_prompt_includes_sender_subject_and_text() -> None:
    prompt = llm_classifier._user_prompt(
        text="Merci de confirmer votre compte.",
        sender="support@paypa1-security.com",
        subject="Action immediate requise",
    )

    assert "Domaine expéditeur: paypa1-security.com" in prompt
    assert "support@" not in prompt
    assert "Objet: Action immediate requise" in prompt
    assert "Merci de confirmer votre compte." in prompt


def test_user_prompt_separates_gateway_context_from_untrusted_content() -> None:
    prompt = llm_classifier._user_prompt(
        text="Vous recevez ce message parce que vous êtes abonné.",
        sender="news@example.fr",
        subject="Fwd: Actualités",
        mail_context=MailContext(
            structured_forward=True,
            outer_sender_authenticated=True,
            subscription_claimed=True,
            recipient_expected=True,
            transactional_evidence=True,
        ),
    )

    assert prompt.index("<CONTEXTE_PASSERELLE>") < prompt.index("<EMAIL_NON_FIABLE>")
    assert "transfert_structure=true" in prompt
    assert "expediteur_externe_authentifie=true" in prompt
    assert "attendu_par_destinataire=true" in prompt
    assert "preuve_transactionnelle=true" in prompt


def test_system_prompt_does_not_equate_bulk_format_with_spam() -> None:
    assert "confirmation d'inscription demandée" in llm_classifier._SYSTEM
    assert "désabonnement" in llm_classifier._SYSTEM
    assert "ne prouvent pas à eux seuls" in llm_classifier._SYSTEM


def test_classify_llm_forwards_sender_and_subject(monkeypatch) -> None:
    captured: dict[str, str | None] = {}

    def fake_tier(
        text: str,
        sender: str | None = None,
        subject: str | None = None,
        mail_context: MailContext | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> LLMResult:
        captured["text"] = text
        captured["sender"] = sender
        captured["subject"] = subject
        assert mail_context is None
        assert timeout_seconds is not None
        assert timeout_seconds > 0
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
    class FakeClient:
        post = staticmethod(fake_post)

    monkeypatch.setattr(llm_classifier, "_http_client", lambda: FakeClient())
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
    class FakeClient:
        post = staticmethod(fake_post)

    monkeypatch.setattr(llm_classifier, "_http_client", lambda: FakeClient())

    response = llm_classifier._resilient_post(
        "test-provider", "https://provider.test"
    )

    assert response.status_code == 400
    assert calls == 1


@pytest.mark.parametrize(
    ("caller", "key_name", "provider"),
    [
        (llm_classifier._call_mistral, "MISTRAL_API_KEY", "mistral"),
        (llm_classifier._call_groq, "GROQ_API_KEY", "groq"),
        (llm_classifier._call_cerebras, "CEREBRAS_API_KEY", "cerebras"),
    ],
)
def test_provider_without_key_is_skipped(monkeypatch, caller, key_name, provider) -> None:  # noqa: ANN001
    monkeypatch.delenv(key_name, raising=False)
    before = llm_classifier.provider_event_snapshot().get((provider, "not_configured"), 0)

    assert caller("message") is None
    assert llm_classifier.provider_event_snapshot()[(provider, "not_configured")] == before + 1


def test_openai_compatible_parses_valid_response(monkeypatch) -> None:
    body = {
        "label": "legitimate",
        "confidence": 0.8,
        "probabilities": {"phishing": 0.1, "spam": 0.1, "legitimate": 0.8},
        "explanation": "Message cohérent.",
    }
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "https://provider.test"),
        json={"choices": [{"message": {"content": json.dumps(body)}}]},
    )
    monkeypatch.setattr(llm_classifier, "_resilient_post", lambda *args, **kwargs: response)

    result = llm_classifier._openai_compatible(
        base_url="https://provider.test",
        api_key="token",
        model="model",
        temperature=0.0,
        text="message",
        sender=None,
        subject=None,
        mail_context=None,
        provider="test",
        timeout_seconds=1.0,
    )

    assert result is not None
    assert result.label == "legitimate"
    assert result.probabilities["legitimate"] == 0.8


def test_openai_compatible_handles_http_and_payload_failures(monkeypatch) -> None:
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://provider.test"),
    )
    monkeypatch.setattr(llm_classifier, "_resilient_post", lambda *args, **kwargs: response)

    assert (
        llm_classifier._openai_compatible(
            base_url="https://provider.test",
            api_key="token",
            model="model",
            temperature=0.0,
            text="message",
            sender=None,
            subject=None,
            mail_context=None,
            provider="test",
            timeout_seconds=1.0,
        )
        is None
    )


def test_parse_response_supports_fences_and_fallback_probabilities() -> None:
    result = llm_classifier._parse_response(
        """```json
        {"label":"uncertain","confidence":0.6,"explanation":"Ambigu."}
        ```""",
        "test",
    )

    assert result is not None
    assert result.label == "uncertain"
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        '{"label":"unsupported","confidence":0.5}',
        '{"label":"spam","confidence":0.9,"probabilities":{"phishing":0.8,"spam":0.1,"legitimate":0.1}}',
    ],
)
def test_parse_response_rejects_invalid_provider_contract(payload: str) -> None:
    assert llm_classifier._parse_response(payload, "test") is None


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"phishing": -1, "spam": 1, "legitimate": 1},
        {"phishing": 0, "spam": 0, "legitimate": 0},
        {"phishing": float("nan"), "spam": 0, "legitimate": 1},
    ],
)
def test_probability_normalization_rejects_invalid_values(raw) -> None:  # noqa: ANN001
    assert llm_classifier._normalize_probabilities(raw) is None


def test_circuit_breaker_opens_and_recovers(monkeypatch) -> None:
    llm_classifier._circuit_failures.clear()
    llm_classifier._circuit_opened_at.clear()
    monkeypatch.setenv("LLM_CIRCUIT_BREAKER_FAILURES", "2")
    monkeypatch.setenv("LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS", "10")

    llm_classifier._record_provider_result("test", success=False, now=1.0)
    assert llm_classifier._circuit_allows("test", 2.0) is True
    llm_classifier._record_provider_result("test", success=False, now=2.0)
    assert llm_classifier._circuit_allows("test", 3.0) is False
    assert llm_classifier._circuit_allows("test", 12.0) is True


def test_resilient_post_raises_last_timeout(monkeypatch) -> None:
    class FakeClient:
        @staticmethod
        def post(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            raise httpx.ReadTimeout("slow")

    llm_classifier._circuit_failures.clear()
    llm_classifier._circuit_opened_at.clear()
    monkeypatch.setenv("LLM_MAX_ATTEMPTS", "1")
    monkeypatch.setattr(llm_classifier, "_http_client", lambda: FakeClient())

    with pytest.raises(httpx.ReadTimeout):
        llm_classifier._resilient_post("test", "https://provider.test")


def test_classify_llm_falls_through_and_reports_unavailable(monkeypatch) -> None:
    calls: list[str] = []

    def unavailable(text: str, **kwargs) -> None:  # noqa: ANN003
        calls.append(text)
        return None

    monkeypatch.setattr(llm_classifier, "_TIERS", (unavailable, unavailable))

    assert llm_classifier.classify_llm("message") is None
    assert calls == ["message", "message"]


def test_environment_parsers_use_defaults_for_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("FLOAT_VALUE", "invalid")
    monkeypatch.setenv("INT_VALUE", "invalid")

    assert llm_classifier._env_float("FLOAT_VALUE", 2.5) == 2.5
    assert llm_classifier._env_int("INT_VALUE", 3) == 3


def test_exception_categories_are_bounded() -> None:
    assert llm_classifier._exception_category(httpx.ReadTimeout("slow")) == "timeout"
    assert (
        llm_classifier._exception_category(httpx.ConnectError("offline"))
        == "connection_failed"
    )
    assert llm_classifier._exception_category(RuntimeError("circuit open")) == "circuit_open"
    assert llm_classifier._exception_category(ValueError("secret detail")) == "request_failed"
