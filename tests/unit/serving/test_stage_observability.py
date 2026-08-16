"""Per-stage observability and bounded provider outcome metrics.

These metrics exist to answer two questions that the previous exposition could
not: which provider is failing and how often, and whether the local model and
the LLM actually disagree. The second is what should drive the fusion weights,
which matters more than usual here because the training corpus is synthetic.
"""

from __future__ import annotations

import pytest

from src.inference import llm_classifier
from src.serving.telemetry import RuntimeTelemetry


@pytest.fixture
def telemetry() -> RuntimeTelemetry:
    return RuntimeTelemetry()


def _observe(telemetry: RuntimeTelemetry, onnx: str | None, llm: str | None) -> None:
    stage_labels = {}
    if onnx:
        stage_labels["onnx"] = onnx
    if llm:
        stage_labels["llm"] = llm
    telemetry.observe(
        status_code=200,
        latency_ms=1200.0,
        verdict="safe",
        label_verdict="legitimate",
        stage_labels=stage_labels,
        mode="llm",
    )


def test_stage_labels_are_counted_per_semantic_stage(telemetry: RuntimeTelemetry) -> None:
    _observe(telemetry, "spam", "legitimate")
    _observe(telemetry, "spam", "spam")

    assert telemetry.stage_label_total[("onnx", "spam")] == 2
    assert telemetry.stage_label_total[("llm", "legitimate")] == 1
    assert telemetry.stage_label_total[("llm", "spam")] == 1


def test_non_semantic_stages_are_not_counted(telemetry: RuntimeTelemetry) -> None:
    """Rules and blocklists emit phishing evidence, not a three-class opinion."""
    telemetry.observe(
        status_code=200,
        latency_ms=10.0,
        stage_labels={"rules": "phishing", "blocklist": "phishing", "onnx": "phishing"},
    )

    assert telemetry.stage_label_total[("onnx", "phishing")] == 1
    assert ("rules", "phishing") not in telemetry.stage_label_total
    assert ("blocklist", "phishing") not in telemetry.stage_label_total


def test_disagreement_between_local_model_and_llm_is_recorded(
    telemetry: RuntimeTelemetry,
) -> None:
    _observe(telemetry, "spam", "legitimate")
    _observe(telemetry, "spam", "legitimate")
    _observe(telemetry, "phishing", "phishing")

    assert telemetry.stage_agreement_total[("spam", "legitimate")] == 2
    assert telemetry.stage_agreement_total[("phishing", "phishing")] == 1

    rendered = telemetry.to_prometheus()
    assert 'onnx_label="spam",llm_label="legitimate",agreement="disagree"' in rendered
    assert 'onnx_label="phishing",llm_label="phishing",agreement="agree"' in rendered


def test_agreement_needs_both_stages(telemetry: RuntimeTelemetry) -> None:
    """A local-only request has nothing to compare and must not be counted."""
    _observe(telemetry, "spam", None)

    assert not telemetry.stage_agreement_total
    assert telemetry.stage_label_total[("onnx", "spam")] == 1


def test_provider_outcomes_are_counted_not_only_logged(
    monkeypatch: pytest.MonkeyPatch, telemetry: RuntimeTelemetry
) -> None:
    """Failure counts need a success denominator to be interpretable."""
    monkeypatch.setattr(llm_classifier, "_provider_events", llm_classifier.Counter())

    llm_classifier._emit_provider_event("mistral", "selected")
    llm_classifier._emit_provider_event("mistral", "timeout")
    llm_classifier._emit_provider_event("groq", "selected")

    snapshot = llm_classifier.provider_event_snapshot()
    assert snapshot[("mistral", "selected")] == 1
    assert snapshot[("mistral", "timeout")] == 1
    assert snapshot[("groq", "selected")] == 1

    rendered = telemetry.to_prometheus()
    assert 'provider="mistral",category="timeout"' in rendered
    assert 'provider="mistral",category="selected"' in rendered


def test_snapshot_is_a_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers must not be able to mutate live counters through the snapshot."""
    monkeypatch.setattr(llm_classifier, "_provider_events", llm_classifier.Counter())
    llm_classifier._emit_provider_event("mistral", "selected")

    snapshot = llm_classifier.provider_event_snapshot()
    snapshot[("mistral", "selected")] = 999

    assert llm_classifier.provider_event_snapshot()[("mistral", "selected")] == 1


def test_metric_cardinality_stays_bounded(telemetry: RuntimeTelemetry) -> None:
    """Guard the free-tier series budget: both metrics are closed-set products."""
    labels = ("phishing", "spam", "legitimate", "uncertain")
    for onnx_label in labels:
        for llm_label in labels:
            _observe(telemetry, onnx_label, llm_label)

    # 4x4 pairs and 2 stages x 4 labels, regardless of traffic volume.
    assert len(telemetry.stage_agreement_total) == 16
    assert len(telemetry.stage_label_total) == 8
