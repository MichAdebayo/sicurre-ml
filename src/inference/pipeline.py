"""Composite inference pipeline — orchestrates all four stages.

Stage weights (configurable via env):
    WEIGHT_RULES      default 0.10
  WEIGHT_BLOCKLIST  default 0.25
    WEIGHT_ONNX       default 0.20
    WEIGHT_LLM        default 0.45

A stage that produces no signal (e.g. no URLs for rules, LLM failure) has
its weight redistributed proportionally to the stages that did run.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from src.inference.blocklist import BlocklistResult, check_blocklists
from src.inference.llm_classifier import LLMResult, classify_llm
from src.inference.onnx_classifier import OnnxResult, classify_onnx
from src.inference.rules import RuleResult, check_url_rules


@dataclass
class ClassificationResult:
    verdict: str                        # "phishing" | "safe"
    label_verdict: str                  # "phishing" | "spam" | "legitimate"
    composite_score: float              # 0.0–1.0  (phishing probability)
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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _phishing_score(label: str, confidence: float) -> float:
    """Convert a label + confidence to a phishing probability in [0, 1]."""
    return confidence if label == "phishing" else 1.0 - confidence


def _onnx_phishing_score(result: OnnxResult) -> float:
    """Map the 3-class ONNX output to a binary phishing probability."""
    phishing_prob = result.raw_scores.get("phishing")
    if phishing_prob is not None:
        return float(phishing_prob)
    return _phishing_score(result.label, result.confidence)


def _normalize_distribution(distribution: dict[str, float]) -> dict[str, float]:
    labels = {"phishing", "spam", "legitimate"}
    merged = {label: max(0.0, float(distribution.get(label, 0.0))) for label in labels}
    total = sum(merged.values())
    if total <= 0.0:
        return {"phishing": 0.0, "spam": 0.5, "legitimate": 0.5}
    return {label: merged[label] / total for label in labels}


def _binary_to_distribution(
    *,
    phishing_prob: float,
    non_phishing_split: tuple[float, float] = (0.5, 0.5),
) -> dict[str, float]:
    p = min(max(phishing_prob, 0.0), 1.0)
    spam_share, legit_share = non_phishing_split
    spread = max(spam_share, 0.0) + max(legit_share, 0.0)
    if spread <= 0.0:
        spam_share, legit_share = 0.5, 0.5
        spread = 1.0
    non_p = 1.0 - p
    return {
        "phishing": p,
        "spam": non_p * (spam_share / spread),
        "legitimate": non_p * (legit_share / spread),
    }


def run_pipeline(
    text: str,
    subject: str | None = None,
    sender: str | None = None,
    use_virustotal: bool = False,
    use_llm: bool = True,
) -> ClassificationResult:
    """Run all stages and return a composite ClassificationResult.

    Parameters
    ----------
    text:
        Raw message text (SMS, email body, etc.)
    subject:
        Optional email subject used by the LLM stage for extra context.
    sender:
        Optional sender email/address used by the LLM stage for extra context.
    use_virustotal:
        Pass True to enable the VirusTotal enrichment in the blocklist stage.
        Adds latency; off by default.
    use_llm:
        Set False to skip the LLM stage (useful in unit tests or when budget
        is a concern for a specific request).
    """
    # Stage weights
    w_rules = _env_float("WEIGHT_RULES", 0.10)
    w_block = _env_float("WEIGHT_BLOCKLIST", 0.25)
    w_onnx = _env_float("WEIGHT_ONNX", 0.20)
    w_llm = _env_float("WEIGHT_LLM", 0.45)

    stage_scores: dict[str, float] = {}
    stage_labels: dict[str, str] = {}
    stage_latencies_ms: dict[str, float] = {}
    active_weights: dict[str, float] = {}
    stage_distributions: dict[str, dict[str, float]] = {}
    stage_breakdown: dict[str, dict[str, Any]] = {}
    explanation = ""
    llm_provider = ""

    # ── Stage 1: Rules ───────────────────────────────────────────────────────
    stage_started = time.perf_counter()
    rule_res: RuleResult = check_url_rules(text)
    stage_latencies_ms["rules"] = round((time.perf_counter() - stage_started) * 1000.0, 3)
    stage_breakdown["rules"] = {
        "active": False,
        "configured_weight": w_rules,
        "reason": "No URLs found",
        "risk_score": rule_res.risk_score,
        "reasons": rule_res.reasons,
    }
    if rule_res.reasons and rule_res.reasons != ["No URLs found"]:
        score = _phishing_score(
            "phishing" if rule_res.is_phishing else "safe",
            rule_res.confidence,
        )
        stage_scores["rules"] = score
        stage_labels["rules"] = "phishing" if rule_res.is_phishing else "safe"
        active_weights["rules"] = w_rules
        stage_distributions["rules"] = _binary_to_distribution(phishing_prob=score)
        stage_breakdown["rules"] = {
            "active": True,
            "configured_weight": w_rules,
            "reason": "URL rule evidence available",
            "risk_score": rule_res.risk_score,
            "reasons": rule_res.reasons,
            "phishing_score": score,
            "label_distribution": stage_distributions["rules"],
        }
    # if no URLs found, rules stage is omitted from weighting

    # ── Stage 2: Blocklist ───────────────────────────────────────────────────
    stage_started = time.perf_counter()
    block_res: BlocklistResult = check_blocklists(text, use_virustotal=use_virustotal)
    stage_latencies_ms["blocklist"] = round((time.perf_counter() - stage_started) * 1000.0, 3)
    stage_breakdown["blocklist"] = {
        "active": False,
        "configured_weight": w_block,
        "source": block_res.source,
        "detail": block_res.detail,
        "reason": "No blocklist hit",
    }
    if block_res.source != "clean" or block_res.is_known_phishing:
        score = _phishing_score(
            "phishing" if block_res.is_known_phishing else "safe",
            block_res.confidence if block_res.is_known_phishing else 0.0,
        )
        stage_scores["blocklist"] = score
        stage_labels["blocklist"] = "phishing" if block_res.is_known_phishing else "safe"
        active_weights["blocklist"] = w_block
        stage_distributions["blocklist"] = _binary_to_distribution(phishing_prob=score)
        stage_breakdown["blocklist"] = {
            "active": True,
            "configured_weight": w_block,
            "source": block_res.source,
            "detail": block_res.detail,
            "reason": "Matched known malicious indicator",
            "phishing_score": score,
            "label_distribution": stage_distributions["blocklist"],
        }
    # A clean blocklist miss is neutral evidence, so it is omitted.

    # ── Stage 3: ONNX ────────────────────────────────────────────────────────
    stage_started = time.perf_counter()
    onnx_res: OnnxResult = classify_onnx(text)
    stage_latencies_ms["onnx"] = round((time.perf_counter() - stage_started) * 1000.0, 3)
    stage_scores["onnx"] = _onnx_phishing_score(onnx_res)
    stage_labels["onnx"] = onnx_res.label
    active_weights["onnx"] = w_onnx
    onnx_distribution = _normalize_distribution(onnx_res.raw_scores)
    stage_distributions["onnx"] = onnx_distribution
    stage_breakdown["onnx"] = {
        "active": True,
        "configured_weight": w_onnx,
        "reason": "Base model output",
        "predicted_label": onnx_res.label,
        "confidence": onnx_res.confidence,
        "phishing_score": stage_scores["onnx"],
        "label_distribution": onnx_distribution,
    }

    # ── Stage 4: LLM ─────────────────────────────────────────────────────────
    llm_res: LLMResult | None = None
    stage_breakdown["llm"] = {
        "active": False,
        "configured_weight": w_llm,
        "reason": "Disabled or no provider response",
    }
    if use_llm:
        stage_started = time.perf_counter()
        llm_res = classify_llm(text, sender=sender, subject=subject)
        stage_latencies_ms["llm"] = round((time.perf_counter() - stage_started) * 1000.0, 3)
    if llm_res is not None:
        stage_scores["llm"] = _phishing_score(llm_res.label, llm_res.confidence)
        stage_labels["llm"] = llm_res.label
        active_weights["llm"] = w_llm
        explanation = llm_res.explanation
        llm_provider = llm_res.provider

        non_phishing_split = (
            onnx_distribution.get("spam", 0.5),
            onnx_distribution.get("legitimate", 0.5),
        )
        stage_distributions["llm"] = _binary_to_distribution(
            phishing_prob=stage_scores["llm"],
            non_phishing_split=non_phishing_split,
        )
        stage_breakdown["llm"] = {
            "active": True,
            "configured_weight": w_llm,
            "reason": "LLM response available",
            "provider": llm_res.provider,
            "label": llm_res.label,
            "confidence": llm_res.confidence,
            "phishing_score": stage_scores["llm"],
            "label_distribution": stage_distributions["llm"],
        }

    # ── Composite score ──────────────────────────────────────────────────────
    total_weight = sum(active_weights.values())
    applied_weights: dict[str, float] = {}
    stage_contributions: dict[str, float] = {}
    if total_weight == 0:
        composite = 0.5  # no signal at all — neutral
        label_distribution = {"phishing": 0.5, "spam": 0.25, "legitimate": 0.25}
    else:
        applied_weights = {
            stage: weight / total_weight for stage, weight in active_weights.items()
        }
        stage_contributions = {
            stage: stage_scores[stage] * applied_weights[stage] for stage in applied_weights
        }
        composite = sum(stage_contributions.values())

        aggregate_distribution = {"phishing": 0.0, "spam": 0.0, "legitimate": 0.0}
        for stage, weight in applied_weights.items():
            dist = stage_distributions.get(stage)
            if dist is None:
                dist = _binary_to_distribution(phishing_prob=stage_scores[stage])
            for label in aggregate_distribution:
                aggregate_distribution[label] += dist[label] * weight
        label_distribution = _normalize_distribution(aggregate_distribution)

    for stage_name, details in stage_breakdown.items():
        details["applied_weight"] = round(applied_weights.get(stage_name, 0.0), 6)
        details["contribution"] = round(stage_contributions.get(stage_name, 0.0), 6)

    threshold = _env_float("PHISHING_THRESHOLD", 0.5)
    is_phishing = composite >= threshold
    verdict = "phishing" if is_phishing else "safe"
    label_verdict = max(
        label_distribution.items(),
        key=lambda item: item[1],
    )[0]

    return ClassificationResult(
        verdict=verdict,
        label_verdict=label_verdict,
        composite_score=round(composite, 4),
        is_phishing=is_phishing,
        stage_scores={k: round(v, 4) for k, v in stage_scores.items()},
        stage_labels=stage_labels,
        label_distribution={k: round(v, 4) for k, v in label_distribution.items()},
        stage_weights_configured={
            "rules": round(w_rules, 4),
            "blocklist": round(w_block, 4),
            "onnx": round(w_onnx, 4),
            "llm": round(w_llm, 4),
        },
        stage_weights_applied={k: round(v, 4) for k, v in applied_weights.items()},
        stage_contributions={k: round(v, 4) for k, v in stage_contributions.items()},
        stage_breakdown=stage_breakdown,
        stage_latencies_ms=stage_latencies_ms,
        explanation=explanation,
        llm_provider=llm_provider,
    )
