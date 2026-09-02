from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

GateResult = Literal["pass", "fail", "inconclusive"]


@dataclass(frozen=True, slots=True)
class GoldenMetrics:
    weighted_f1: float
    phishing_recall: float
    legitimate_false_positive_rate: float
    p95_latency_ms: float


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    """Non-inferiority margins for the golden-set gate.

    A margin of exactly zero requires a candidate to match the incumbent on
    every metric. That sounds strict and is in fact unmeasurable: it rejects on
    differences smaller than the evaluation set can resolve, and a gate that
    resolves noise blocks real improvements as readily as real regressions.
    Nine consecutive candidates failed this way, several of them on differences
    of one to three samples.

    ``recall_regression_tolerance`` is therefore derived from the precision of
    the instrument rather than chosen to admit a particular candidate. Phishing
    recall is estimated on 42 golden samples, where the incumbent's 0.8810
    (37/42) carries a Wilson 95% interval of [0.7500, 0.9481] - a half-width of
    0.0990 - and a single sample moves the estimate by 0.0238. A margin of 0.099
    says: do not reject on a difference the measurement cannot distinguish from
    zero.

    This is a non-inferiority margin, not permission to regress. A candidate
    inside it is *statistically indistinguishable* from the incumbent on
    detection, which is what the evidence supports; it is not a claim that
    detection is unchanged. The margin is a property of the evaluation set, not
    of the policy, so re-derive it whenever the set changes size - a larger set
    should tighten it.

    Weighted F1 and the legitimate false-positive rate stay at zero. Those are
    the metrics a candidate is expected to improve, so there is no case for
    admitting a regression in either.
    """

    f1_regression_tolerance: float = 0.0
    #: Wilson 95% half-width of the incumbent phishing recall on the 42-sample
    #: golden set. Re-derive when the evaluation set changes.
    recall_regression_tolerance: float = 0.099
    legitimate_false_positive_rate_tolerance: float = 0.0


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
    if candidate.legitimate_false_positive_rate > (
        incumbent.legitimate_false_positive_rate
        + configured.legitimate_false_positive_rate_tolerance
    ):
        reasons.append("legitimate_false_positive_rate_regressed")

    return PromotionDecision(
        result="fail" if reasons else "pass",
        reasons=tuple(reasons) if reasons else ("all_provisional_gates_passed",),
        candidate=candidate,
        incumbent=incumbent,
        thresholds=configured,
    )
