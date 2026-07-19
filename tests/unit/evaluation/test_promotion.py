from __future__ import annotations

from src.evaluation.promotion import (
    GoldenMetrics,
    PromotionThresholds,
    decide_candidate_promotion,
)


def _metrics(**overrides: float) -> GoldenMetrics:
    values = {
        "weighted_f1": 0.95,
        "phishing_recall": 0.98,
        "legitimate_false_positive_rate": 0.01,
        "p95_latency_ms": 350.0,
    }
    values.update(overrides)
    return GoldenMetrics(**values)


def test_pass_still_requires_manual_approval() -> None:
    decision = decide_candidate_promotion(_metrics(), _metrics(weighted_f1=0.94))

    assert decision.result == "pass"
    assert decision.requires_manual_approval is True


def test_rejects_recall_and_legitimate_false_positive_regression() -> None:
    decision = decide_candidate_promotion(
        _metrics(phishing_recall=0.90, legitimate_false_positive_rate=0.05),
        _metrics(),
    )

    assert decision.result == "fail"
    assert "phishing_recall_regressed" in decision.reasons
    assert "legitimate_false_positive_rate_regressed" in decision.reasons


def test_latency_is_diagnostic_and_does_not_block() -> None:
    decision = decide_candidate_promotion(
        _metrics(p95_latency_ms=50_000),
        _metrics(p95_latency_ms=100),
    )

    assert decision.result == "pass"


def test_missing_incumbent_is_inconclusive() -> None:
    decision = decide_candidate_promotion(
        _metrics(),
        None,
        PromotionThresholds(),
    )

    assert decision.result == "inconclusive"
