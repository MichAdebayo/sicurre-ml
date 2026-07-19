from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

GateResult = Literal["pass", "fail", "inconclusive"]


@dataclass(frozen=True, slots=True)
class GoldenMetrics:
    weighted_f1: float
    phishing_recall: float
    phishing_false_positive_rate: float
    p95_latency_ms: float


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    f1_regression_tolerance: float = 0.0
    recall_regression_tolerance: float = 0.0
    false_positive_rate_tolerance: float = 0.0
    p95_latency_ceiling_ms: float = 1000.0


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    result: GateResult
    reasons: tuple[str, ...]
    candidate: GoldenMetrics | None
    incumbent: GoldenMetrics | None
    thresholds: PromotionThresholds
    requires_manual_approval: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def decide_candidate_promotion(
    candidate: GoldenMetrics | None,
    incumbent: GoldenMetrics | None,
    thresholds: PromotionThresholds | None = None,
) -> PromotionDecision:
    configured = thresholds or PromotionThresholds()
    if candidate is None or incumbent is None:
        return PromotionDecision(
            result="inconclusive",
            reasons=("candidate_and_incumbent_metrics_are_required",),
            candidate=candidate,
            incumbent=incumbent,
            thresholds=configured,
        )

    reasons: list[str] = []
    if candidate.weighted_f1 < (
        incumbent.weighted_f1 - configured.f1_regression_tolerance
    ):
        reasons.append("weighted_f1_regressed")
    if candidate.phishing_recall < (
        incumbent.phishing_recall - configured.recall_regression_tolerance
    ):
        reasons.append("phishing_recall_regressed")
    if candidate.phishing_false_positive_rate > (
        incumbent.phishing_false_positive_rate
        + configured.false_positive_rate_tolerance
    ):
        reasons.append("phishing_false_positive_rate_regressed")
    if candidate.p95_latency_ms > configured.p95_latency_ceiling_ms:
        reasons.append("p95_latency_ceiling_exceeded")

    return PromotionDecision(
        result="fail" if reasons else "pass",
        reasons=tuple(reasons) if reasons else ("all_provisional_gates_passed",),
        candidate=candidate,
        incumbent=incumbent,
        thresholds=configured,
    )
