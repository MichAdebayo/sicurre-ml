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
        lambda text: LLMResult(
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