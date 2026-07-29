from __future__ import annotations

from src.inference.blocklist import BlocklistResult
from src.inference.llm_classifier import LLMResult
from src.inference.mail_context import MailContext
from src.inference.onnx_classifier import OnnxResult
from src.inference.pipeline import run_pipeline
from src.inference.rules import RuleResult


def test_pipeline_uses_phishing_probability_and_skips_clean_blocklist(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.inference.pipeline.check_url_rules",
        lambda text: RuleResult(is_phishing=False, confidence=0.0, reasons=["No URLs found"]),
    )
    monkeypatch.setattr(
        "src.inference.pipeline.check_blocklists",
        lambda text, use_virustotal=False: BlocklistResult(
            is_known_phishing=False,
            confidence=0.0,
            source="clean",
        ),
    )
    monkeypatch.setattr(
        "src.inference.pipeline.classify_onnx",
        lambda text: OnnxResult(
            label="spam",
            confidence=0.8,
            raw_scores={"phishing": 0.1, "spam": 0.8, "legitimate": 0.1},
        ),
    )
    monkeypatch.setattr(
        "src.inference.pipeline.classify_llm",
        lambda text, sender=None, subject=None, mail_context=None: LLMResult(
            label="phishing",
            confidence=0.95,
            explanation="Suspicious urgency.",
            provider="groq",
        ),
    )

    result = run_pipeline("message", use_llm=True)

    assert result.is_phishing is True
    assert result.stage_scores["onnx"] == 0.1
    assert "blocklist" not in result.stage_scores
    assert result.llm_provider == "groq"
    assert result.verdict == "phishing"
    assert result.label_verdict in {"phishing", "spam", "legitimate"}
    assert set(result.label_distribution) == {"phishing", "spam", "legitimate"}
    assert round(sum(result.label_distribution.values()), 4) == 1.0
    assert result.stage_weights_applied["onnx"] > 0
    assert result.stage_weights_applied["llm"] > 0
    assert result.stage_weights_applied.get("blocklist", 0.0) == 0.0
    assert result.stage_breakdown["blocklist"]["active"] is False
    assert result.stage_breakdown["llm"]["active"] is True


def test_pipeline_forwards_sender_and_subject_to_llm(monkeypatch) -> None:
    captured: dict[str, str | None] = {}

    monkeypatch.setattr(
        "src.inference.pipeline.check_url_rules",
        lambda text: RuleResult(is_phishing=False, confidence=0.0, reasons=["No URLs found"]),
    )
    monkeypatch.setattr(
        "src.inference.pipeline.check_blocklists",
        lambda text, use_virustotal=False: BlocklistResult(
            is_known_phishing=False,
            confidence=0.0,
            source="clean",
        ),
    )
    monkeypatch.setattr(
        "src.inference.pipeline.classify_onnx",
        lambda text: OnnxResult(
            label="legitimate",
            confidence=0.9,
            raw_scores={"phishing": 0.05, "spam": 0.1, "legitimate": 0.85},
        ),
    )

    def fake_llm(
        text: str,
        sender: str | None = None,
        subject: str | None = None,
        mail_context: MailContext | None = None,
    ) -> LLMResult:
        captured["sender"] = sender
        captured["subject"] = subject
        assert mail_context == MailContext()
        return LLMResult(
            label="legitimate",
            confidence=0.9,
            explanation="No obvious phishing indicators.",
            provider="groq",
            probabilities={"phishing": 0.02, "spam": 0.08, "legitimate": 0.9},
        )

    monkeypatch.setattr("src.inference.pipeline.classify_llm", fake_llm)

    run_pipeline(
        text="Bonjour, voici votre facture.",
        sender="support@paypa1-security.com",
        subject="Action immediate requise",
        use_llm=True,
    )

    assert captured["sender"] == "paypa1-security.com"
    assert captured["subject"] == "Action immediate requise"


def test_llm_can_correct_local_spam_to_legitimate(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.inference.pipeline.check_url_rules",
        lambda text: RuleResult(is_phishing=False, confidence=0.0, reasons=[]),
    )
    monkeypatch.setattr(
        "src.inference.pipeline.check_blocklists",
        lambda text, use_virustotal=False: BlocklistResult(
            is_known_phishing=False,
            confidence=0.0,
            source="clean",
        ),
    )
    monkeypatch.setattr(
        "src.inference.pipeline.classify_onnx",
        lambda text: OnnxResult(
            label="spam",
            confidence=0.8,
            raw_scores={"phishing": 0.05, "spam": 0.8, "legitimate": 0.15},
        ),
    )
    monkeypatch.setattr(
        "src.inference.pipeline.classify_llm",
        lambda text, sender=None, subject=None, mail_context=None: LLMResult(
            label="legitimate",
            confidence=0.95,
            explanation="Message attendu et cohérent.",
            provider="mistral",
            probabilities={"phishing": 0.01, "spam": 0.04, "legitimate": 0.95},
        ),
    )

    result = run_pipeline("Confirmation de votre inscription.", use_llm=True)

    assert result.verdict == "safe"
    assert result.label_verdict == "legitimate"
    assert result.label_distribution["legitimate"] > result.label_distribution["spam"]


def test_missing_llm_is_explicitly_degraded(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.inference.pipeline.check_url_rules",
        lambda text: RuleResult(is_phishing=False, confidence=0.0, reasons=[]),
    )
    monkeypatch.setattr(
        "src.inference.pipeline.check_blocklists",
        lambda text, use_virustotal=False: BlocklistResult(
            is_known_phishing=False,
            confidence=0.0,
            source="clean",
        ),
    )
    monkeypatch.setattr(
        "src.inference.pipeline.classify_onnx",
        lambda text: OnnxResult(
            label="legitimate",
            confidence=0.9,
            raw_scores={"phishing": 0.02, "spam": 0.08, "legitimate": 0.9},
        ),
    )
    monkeypatch.setattr(
        "src.inference.pipeline.classify_llm",
        lambda text, sender=None, subject=None, mail_context=None: None,
    )

    result = run_pipeline("Message attendu.", use_llm=True)

    assert result.degraded_reasons == ["llm_unavailable"]
    assert result.stage_breakdown["llm"]["active"] is False


def test_uncertain_llm_distribution_is_shrunk_toward_neutral(monkeypatch) -> None:
    from src.inference.pipeline import _distribution_from_result

    monkeypatch.setenv("LLM_UNCERTAIN_EVIDENCE_FACTOR", "0.35")
    result = _distribution_from_result(
        LLMResult(
            label="uncertain",
            confidence=0.8,
            explanation="Indices contradictoires.",
            provider="mistral",
            probabilities={"phishing": 0.1, "spam": 0.1, "legitimate": 0.8},
        )
    )

    assert result["legitimate"] < 0.5
    assert result["phishing"] > 0.2
    assert round(sum(result.values()), 8) == 1.0


def _stub_promotional_semantics(monkeypatch, *, known_phishing: bool = False) -> None:
    monkeypatch.setattr(
        "src.inference.pipeline.check_url_rules",
        lambda text: RuleResult(is_phishing=False, confidence=0.0, reasons=[]),
    )
    monkeypatch.setattr(
        "src.inference.pipeline.check_blocklists",
        lambda text, use_virustotal=False: BlocklistResult(
            is_known_phishing=known_phishing,
            confidence=0.99 if known_phishing else 0.0,
            source="phishtank" if known_phishing else "clean",
        ),
    )
    monkeypatch.setattr(
        "src.inference.pipeline.classify_onnx",
        lambda text: OnnxResult(
            label="spam",
            confidence=0.8,
            raw_scores={"phishing": 0.05, "spam": 0.8, "legitimate": 0.15},
        ),
    )
    monkeypatch.setattr(
        "src.inference.pipeline.classify_llm",
        lambda text, sender=None, subject=None, mail_context=None: LLMResult(
            label="spam",
            confidence=0.95,
            explanation="Contenu promotionnel.",
            provider="mistral",
            probabilities={"phishing": 0.01, "spam": 0.95, "legitimate": 0.04},
        ),
    )


def test_structured_forward_resolves_low_risk_newsletter_as_legitimate(monkeypatch) -> None:
    _stub_promotional_semantics(monkeypatch)

    result = run_pipeline(
        "Newsletter transférée avec lien propre.",
        subject="Fwd: Newsletter",
        sender="friend@example.fr",
        mail_context=MailContext(
            structured_forward=True,
            outer_sender_authenticated=True,
            subscription_claimed=True,
        ),
    )

    assert result.verdict == "safe"
    assert result.label_verdict == "legitimate"
    assert result.stage_breakdown["mail_context"]["active"] is True
    assert result.label_distribution["legitimate"] > result.label_distribution["spam"]


def test_subscription_claim_without_forward_does_not_self_whitelist(monkeypatch) -> None:
    _stub_promotional_semantics(monkeypatch)

    result = run_pipeline(
        "Vous recevez ce message parce que vous êtes abonné.",
        mail_context=MailContext(
            mailing_list_headers=True,
            subscription_claimed=True,
        ),
    )

    assert result.label_verdict == "spam"
    assert result.stage_breakdown["mail_context"]["active"] is False


def test_structured_forward_cannot_override_known_malicious_url(monkeypatch) -> None:
    _stub_promotional_semantics(monkeypatch, known_phishing=True)

    result = run_pipeline(
        "Message transféré contenant https://malicious.example/login",
        subject="Fwd: Newsletter",
        mail_context=MailContext(
            structured_forward=True,
            outer_sender_authenticated=True,
        ),
    )

    assert result.verdict == "phishing"
    assert result.label_verdict == "phishing"
    assert result.stage_breakdown["blocklist"]["active"] is True
