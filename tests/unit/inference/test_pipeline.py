from __future__ import annotations

from src.inference.blocklist import BlocklistResult
from src.inference.llm_classifier import LLMResult
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
        lambda text, sender=None, subject=None: LLMResult(
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

    def fake_llm(text: str, sender: str | None = None, subject: str | None = None) -> LLMResult:
        captured["sender"] = sender
        captured["subject"] = subject
        return LLMResult(
            label="safe",
            confidence=0.9,
            explanation="No obvious phishing indicators.",
            provider="groq",
        )

    monkeypatch.setattr("src.inference.pipeline.classify_llm", fake_llm)

    run_pipeline(
        text="Bonjour, voici votre facture.",
        sender="support@paypa1-security.com",
        subject="Action immediate requise",
        use_llm=True,
    )

    assert captured["sender"] == "support@paypa1-security.com"
    assert captured["subject"] == "Action immediate requise"