"""Composite inference pipeline with three-class semantic fusion.

The local ONNX model and external LLM are semantic classifiers. URL rules and
blocklists are independent phishing evidence and can raise phishing risk
without inventing a spam-versus-legitimate opinion.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from src.inference.blocklist import BlocklistResult, check_blocklists
from src.inference.input_normalizer import CanonicalEmail, canonicalize_email
from src.inference.llm_classifier import LLMResult, classify_llm
from src.inference.onnx_classifier import OnnxResult, classify_onnx
from src.inference.rules import RuleResult, check_url_rules

_LABELS = ("phishing", "spam", "legitimate")
_DEFAULT_RULE_WEIGHT = 0.10
_DEFAULT_BLOCKLIST_WEIGHT = 0.25
_SEMANTIC_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="sicurre-semantic",
)


@dataclass
class ClassificationResult:
    verdict: str
    label_verdict: str
    composite_score: float
    is_phishing: bool
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)
    stage_scores: dict[str, float] = field(default_factory=dict)
    stage_labels: dict[str, str] = field(default_factory=dict)
    label_distribution: dict[str, float] = field(default_factory=dict)
    stage_weights_configured: dict[str, float] = field(default_factory=dict)
    stage_weights_applied: dict[str, float] = field(default_factory=dict)
    stage_contributions: dict[str, float] = field(default_factory=dict)
    stage_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    explanation: str = ""
    llm_provider: str = ""
    degraded_reasons: list[str] = field(default_factory=list)
    input_format: str = "plain"


def close_pipeline_resources() -> None:
    """Release shared inference worker threads during process shutdown."""

    _SEMANTIC_EXECUTOR.shutdown(wait=False, cancel_futures=True)


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return value if math.isfinite(value) and value >= 0.0 else default
    except ValueError:
        return default


def _normalize_distribution(distribution: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for label in _LABELS:
        try:
            value = float(distribution.get(label, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        merged[label] = value if math.isfinite(value) and value > 0.0 else 0.0
    total = sum(merged.values())
    if total <= 0.0:
        return {"phishing": 1.0 / 3.0, "spam": 1.0 / 3.0, "legitimate": 1.0 / 3.0}
    return {label: merged[label] / total for label in _LABELS}


def _rounded_distribution(distribution: dict[str, float]) -> dict[str, float]:
    """Round a distribution while preserving the exact probability invariant."""

    normalized = _normalize_distribution(distribution)
    rounded = {label: round(normalized[label], 4) for label in _LABELS}
    residual = round(1.0 - sum(rounded.values()), 4)
    largest = max(_LABELS, key=lambda label: rounded[label])
    rounded[largest] = round(rounded[largest] + residual, 4)
    return rounded


def _distribution_from_result(result: LLMResult) -> dict[str, float]:
    if result.probabilities:
        return _normalize_distribution(result.probabilities)
    if result.label in _LABELS:
        confidence = min(max(result.confidence, 0.0), 1.0)
        residual = (1.0 - confidence) / 2.0
        return {
            label: confidence if label == result.label else residual
            for label in _LABELS
        }
    return {label: 1.0 / 3.0 for label in _LABELS}


def _fuse_semantic_distributions(
    stages: dict[str, dict[str, float]],
    weights: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    active = {
        stage: max(weights.get(stage, 0.0), 0.0)
        for stage in stages
        if weights.get(stage, 0.0) > 0.0
    }
    total = sum(active.values())
    if total <= 0.0:
        return (
            {"phishing": 1.0 / 3.0, "spam": 1.0 / 3.0, "legitimate": 1.0 / 3.0},
            {},
        )
    applied = {stage: weight / total for stage, weight in active.items()}
    fused = {label: 0.0 for label in _LABELS}
    for stage, weight in applied.items():
        distribution = _normalize_distribution(stages[stage])
        for label in _LABELS:
            fused[label] += distribution[label] * weight
    return _normalize_distribution(fused), applied


def _scale_evidence(score: float, configured: float, default: float) -> float:
    bounded = min(max(score, 0.0), 1.0)
    if bounded <= 0.0 or configured <= 0.0:
        return 0.0
    ratio = min(max(configured / default, 0.0), 2.0)
    return 1.0 - (1.0 - bounded) ** ratio


def _apply_phishing_evidence(
    distribution: dict[str, float],
    evidence: list[tuple[str, float]],
) -> dict[str, float]:
    result = _normalize_distribution(distribution)
    phishing = result["phishing"]
    for _, strength in evidence:
        phishing = 1.0 - (1.0 - phishing) * (1.0 - min(max(strength, 0.0), 1.0))
    phishing = min(max(phishing, 0.0), 1.0)
    non_phishing = max(1.0 - result["phishing"], 1e-12)
    spam_share = result["spam"] / non_phishing
    legitimate_share = result["legitimate"] / non_phishing
    remaining = 1.0 - phishing
    return _normalize_distribution(
        {
            "phishing": phishing,
            "spam": remaining * spam_share,
            "legitimate": remaining * legitimate_share,
        }
    )


def _timed_onnx(canonical: CanonicalEmail) -> tuple[OnnxResult, float]:
    started = time.perf_counter()
    result = classify_onnx(canonical.model_text)
    return result, (time.perf_counter() - started) * 1000.0


def _timed_llm(canonical: CanonicalEmail) -> tuple[LLMResult | None, float]:
    started = time.perf_counter()
    result = classify_llm(
        canonical.body,
        sender=canonical.sender_domain,
        subject=canonical.subject,
    )
    return result, (time.perf_counter() - started) * 1000.0


def run_pipeline(
    text: str,
    subject: str | None = None,
    sender: str | None = None,
    use_virustotal: bool = False,
    use_llm: bool = True,
) -> ClassificationResult:
    """Run canonicalization, semantic classification and phishing evidence fusion."""

    w_rules = _env_float("WEIGHT_RULES", _DEFAULT_RULE_WEIGHT)
    w_block = _env_float("WEIGHT_BLOCKLIST", _DEFAULT_BLOCKLIST_WEIGHT)
    w_onnx = _env_float("WEIGHT_ONNX", 0.20)
    w_llm = _env_float("WEIGHT_LLM", 0.45)
    threshold = _env_float("PHISHING_THRESHOLD", 0.5)

    canonical = canonicalize_email(text, subject=subject, sender=sender)
    stage_latencies_ms: dict[str, float] = {}
    stage_scores: dict[str, float] = {}
    stage_labels: dict[str, str] = {}
    stage_distributions: dict[str, dict[str, float]] = {}
    stage_breakdown: dict[str, dict[str, Any]] = {
        "input": {
            "active": True,
            "source_format": canonical.source_format,
            "body_chars": len(canonical.body),
        }
    }
    degraded_reasons: list[str] = []

    onnx_future: Future[tuple[OnnxResult, float]] = _SEMANTIC_EXECUTOR.submit(
        _timed_onnx,
        canonical,
    )
    llm_future: Future[tuple[LLMResult | None, float]] | None = None
    if use_llm:
        llm_future = _SEMANTIC_EXECUTOR.submit(_timed_llm, canonical)

    started = time.perf_counter()
    rule_result: RuleResult = check_url_rules(canonical.security_text)
    stage_latencies_ms["rules"] = round((time.perf_counter() - started) * 1000.0, 3)
    rule_strength = 0.0
    if rule_result.risk_score > 0:
        rule_strength = _scale_evidence(
            rule_result.confidence,
            w_rules,
            _DEFAULT_RULE_WEIGHT,
        )
        stage_scores["rules"] = rule_strength
        stage_labels["rules"] = "phishing" if rule_result.is_phishing else "suspicious"
    stage_breakdown["rules"] = {
        "active": rule_strength > 0.0,
        "configured_weight": w_rules,
        "risk_score": rule_result.risk_score,
        "reasons": rule_result.reasons,
        "phishing_evidence": round(rule_strength, 6),
    }

    started = time.perf_counter()
    block_result: BlocklistResult = check_blocklists(
        canonical.security_text,
        use_virustotal=use_virustotal,
    )
    stage_latencies_ms["blocklist"] = round((time.perf_counter() - started) * 1000.0, 3)
    block_strength = 0.0
    if block_result.is_known_phishing:
        block_strength = _scale_evidence(
            block_result.confidence,
            w_block,
            _DEFAULT_BLOCKLIST_WEIGHT,
        )
        stage_scores["blocklist"] = block_strength
        stage_labels["blocklist"] = "phishing"
    stage_breakdown["blocklist"] = {
        "active": block_strength > 0.0,
        "configured_weight": w_block,
        "source": block_result.source,
        "detail": block_result.detail,
        "phishing_evidence": round(block_strength, 6),
    }

    onnx_result, onnx_latency = onnx_future.result()
    stage_latencies_ms["onnx"] = round(onnx_latency, 3)
    onnx_distribution = _normalize_distribution(onnx_result.raw_scores)
    stage_distributions["onnx"] = onnx_distribution
    stage_scores["onnx"] = onnx_distribution["phishing"]
    stage_labels["onnx"] = onnx_result.label
    stage_breakdown["onnx"] = {
        "active": True,
        "configured_weight": w_onnx,
        "predicted_label": onnx_result.label,
        "confidence": onnx_result.confidence,
        "label_distribution": onnx_distribution,
    }

    llm_result: LLMResult | None = None
    if llm_future is not None:
        llm_result, llm_latency = llm_future.result()
        stage_latencies_ms["llm"] = round(llm_latency, 3)
    if llm_result is not None:
        llm_distribution = _distribution_from_result(llm_result)
        stage_distributions["llm"] = llm_distribution
        stage_scores["llm"] = llm_distribution["phishing"]
        stage_labels["llm"] = llm_result.label
        stage_breakdown["llm"] = {
            "active": True,
            "configured_weight": w_llm,
            "provider": llm_result.provider,
            "predicted_label": llm_result.label,
            "confidence": llm_result.confidence,
            "label_distribution": llm_distribution,
        }
    else:
        stage_breakdown["llm"] = {
            "active": False,
            "configured_weight": w_llm,
            "reason": "disabled" if not use_llm else "provider_chain_unavailable",
        }
        if use_llm:
            degraded_reasons.append("llm_unavailable")

    semantic_distribution, semantic_weights = _fuse_semantic_distributions(
        stage_distributions,
        {"onnx": w_onnx, "llm": w_llm},
    )
    evidence = [
        (name, strength)
        for name, strength in (
            ("rules", rule_strength),
            ("blocklist", block_strength),
        )
        if strength > 0.0
    ]
    label_distribution = _apply_phishing_evidence(semantic_distribution, evidence)
    phishing_probability = label_distribution["phishing"]
    is_phishing = phishing_probability >= threshold
    verdict = "phishing" if is_phishing else "safe"
    if is_phishing:
        label_verdict = "phishing"
    else:
        label_verdict = max(
            ("spam", "legitimate"),
            key=lambda label: label_distribution[label],
        )

    applied_weights = dict(semantic_weights)
    if rule_strength > 0.0:
        applied_weights["rules"] = rule_strength
    if block_strength > 0.0:
        applied_weights["blocklist"] = block_strength
    stage_contributions = {
        stage: (
            stage_scores.get(stage, 0.0) * semantic_weights[stage]
            if stage in semantic_weights
            else stage_scores.get(stage, 0.0)
        )
        for stage in applied_weights
    }
    for stage, details in stage_breakdown.items():
        details["applied_weight"] = round(applied_weights.get(stage, 0.0), 6)
        details["contribution"] = round(stage_contributions.get(stage, 0.0), 6)

    return ClassificationResult(
        verdict=verdict,
        label_verdict=label_verdict,
        composite_score=round(phishing_probability, 4),
        is_phishing=is_phishing,
        stage_scores={key: round(value, 4) for key, value in stage_scores.items()},
        stage_labels=stage_labels,
        label_distribution=_rounded_distribution(label_distribution),
        stage_weights_configured={
            "rules": round(w_rules, 4),
            "blocklist": round(w_block, 4),
            "onnx": round(w_onnx, 4),
            "llm": round(w_llm, 4),
        },
        stage_weights_applied={
            key: round(value, 4)
            for key, value in applied_weights.items()
        },
        stage_contributions={
            key: round(value, 4)
            for key, value in stage_contributions.items()
        },
        stage_breakdown=stage_breakdown,
        stage_latencies_ms=stage_latencies_ms,
        explanation=llm_result.explanation if llm_result else "",
        llm_provider=llm_result.provider if llm_result else "",
        degraded_reasons=degraded_reasons,
        input_format=canonical.source_format,
    )
