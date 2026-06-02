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
from dataclasses import dataclass, field

from src.inference.blocklist import BlocklistResult, check_blocklists
from src.inference.llm_classifier import LLMResult, classify_llm
from src.inference.onnx_classifier import OnnxResult, classify_onnx
from src.inference.rules import RuleResult, check_url_rules


@dataclass
class ClassificationResult:
    verdict: str                        # "phishing" | "safe"
    composite_score: float              # 0.0–1.0  (phishing probability)
    is_phishing: bool
    stage_scores: dict[str, float] = field(default_factory=dict)
    stage_labels: dict[str, str] = field(default_factory=dict)
    explanation: str = ""
    llm_provider: str = ""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _phishing_score(label: str, confidence: float) -> float:
    """Convert a label + confidence to a phishing probability in [0, 1]."""
    if label == "phishing":
        return confidence
    return 1.0 - confidence


def _onnx_phishing_score(result: OnnxResult) -> float:
    """Map the 3-class ONNX output to a binary phishing probability."""
    phishing_prob = result.raw_scores.get("phishing")
    if phishing_prob is not None:
        return float(phishing_prob)
    return _phishing_score(result.label, result.confidence)


def run_pipeline(
    text: str,
    use_virustotal: bool = False,
    use_llm: bool = True,
) -> ClassificationResult:
    """Run all stages and return a composite ClassificationResult.

    Parameters
    ----------
    text:
        Raw message text (SMS, email body, etc.)
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
    active_weights: dict[str, float] = {}
    explanation = ""
    llm_provider = ""

    # ── Stage 1: Rules ───────────────────────────────────────────────────────
    rule_res: RuleResult = check_url_rules(text)
    if rule_res.reasons and rule_res.reasons != ["No URLs found"]:
        score = _phishing_score(
            "phishing" if rule_res.is_phishing else "safe",
            rule_res.confidence,
        )
        stage_scores["rules"] = score
        stage_labels["rules"] = "phishing" if rule_res.is_phishing else "safe"
        active_weights["rules"] = w_rules
    # if no URLs found, rules stage is omitted from weighting

    # ── Stage 2: Blocklist ───────────────────────────────────────────────────
    block_res: BlocklistResult = check_blocklists(text, use_virustotal=use_virustotal)
    if block_res.source != "clean" or block_res.is_known_phishing:
        score = _phishing_score(
            "phishing" if block_res.is_known_phishing else "safe",
            block_res.confidence if block_res.is_known_phishing else 0.0,
        )
        stage_scores["blocklist"] = score
        stage_labels["blocklist"] = "phishing" if block_res.is_known_phishing else "safe"
        active_weights["blocklist"] = w_block
    # A clean blocklist miss is neutral evidence, so it is omitted.

    # ── Stage 3: ONNX ────────────────────────────────────────────────────────
    onnx_res: OnnxResult = classify_onnx(text)
    stage_scores["onnx"] = _onnx_phishing_score(onnx_res)
    stage_labels["onnx"] = onnx_res.label
    active_weights["onnx"] = w_onnx

    # ── Stage 4: LLM ─────────────────────────────────────────────────────────
    llm_res: LLMResult | None = classify_llm(text) if use_llm else None
    if llm_res is not None:
        stage_scores["llm"] = _phishing_score(llm_res.label, llm_res.confidence)
        stage_labels["llm"] = llm_res.label
        active_weights["llm"] = w_llm
        explanation = llm_res.explanation
        llm_provider = llm_res.provider

    # ── Composite score ──────────────────────────────────────────────────────
    total_weight = sum(active_weights.values())
    if total_weight == 0:
        composite = 0.5  # no signal at all — neutral
    else:
        composite = sum(
            stage_scores[s] * active_weights[s] for s in active_weights
        ) / total_weight

    threshold = _env_float("PHISHING_THRESHOLD", 0.5)
    is_phishing = composite >= threshold
    verdict = "phishing" if is_phishing else "safe"

    return ClassificationResult(
        verdict=verdict,
        composite_score=round(composite, 4),
        is_phishing=is_phishing,
        stage_scores={k: round(v, 4) for k, v in stage_scores.items()},
        stage_labels=stage_labels,
        explanation=explanation,
        llm_provider=llm_provider,
    )
