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
    """A recall drop larger than the non-inferiority margin still fails.

    The margin is 0.099, so the drop here (0.98 -> 0.85) is deliberately well
    outside it. Using a drop inside the margin would make this test pass for the
    wrong reason - the false-positive regression alone would fail the gate.
    """
    decision = decide_candidate_promotion(
        _metrics(phishing_recall=0.85, legitimate_false_positive_rate=0.05),
        _metrics(),
    )

    assert decision.result == "fail"
    assert "phishing_recall_regressed" in decision.reasons
    assert "legitimate_false_positive_rate_regressed" in decision.reasons


def test_recall_drop_inside_the_margin_does_not_block() -> None:
    """A drop the evaluation set cannot resolve is not treated as a regression.

    This is the behaviour the margin exists for: 0.98 -> 0.90 is 0.08, inside
    the 0.099 Wilson half-width of a 42-sample recall estimate. Everything else
    is held equal so the only thing under test is the recall arm.
    """
    decision = decide_candidate_promotion(
        _metrics(phishing_recall=0.90),
        _metrics(),
    )

    assert decision.result == "pass"
    assert "phishing_recall_regressed" not in decision.reasons


def test_margin_applies_only_to_recall() -> None:
    """Weighted F1 and legitimate false positives keep a zero margin.

    They are the metrics a candidate is expected to improve, so an equivalent
    drop in either must still fail even though the same size drop in recall
    would be admitted.
    """
    f1 = decide_candidate_promotion(_metrics(weighted_f1=0.95 - 0.08), _metrics())
    fp = decide_candidate_promotion(
        _metrics(legitimate_false_positive_rate=0.01 + 0.08), _metrics()
    )

    assert "weighted_f1_regressed" in f1.reasons
    assert "legitimate_false_positive_rate_regressed" in fp.reasons


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
