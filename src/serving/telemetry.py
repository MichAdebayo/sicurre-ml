from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


_PROMETHEUS_BUCKETS_MS = (25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0)


def _env_text(name: str, default: str = "unknown") -> str:
    return os.getenv(name, default).strip() or default


def _sanitize_distribution(distribution: dict[str, float]) -> dict[str, float]:
    labels = ("phishing", "spam", "legitimate")
    return {label: float(distribution.get(label, 0.0)) for label in labels}


@dataclass
class RuntimeTelemetry:
    lock: Lock = field(default_factory=Lock)
    request_total: int = 0
    request_status_total: Counter[str] = field(default_factory=Counter)
    verdict_total: Counter[str] = field(default_factory=Counter)
    label_total: Counter[str] = field(default_factory=Counter)
    llm_provider_total: Counter[str] = field(default_factory=Counter)
    error_total: Counter[str] = field(default_factory=Counter)
    total_latency_ms_sum: float = 0.0
    total_latency_ms_count: int = 0
    total_latency_ms_max: float = 0.0
    stage_latency_ms_sum: Counter[str] = field(default_factory=Counter)
    stage_latency_ms_count: Counter[str] = field(default_factory=Counter)
    stage_latency_ms_max: dict[str, float] = field(default_factory=dict)
    label_distribution_total: Counter[str] = field(default_factory=Counter)
    total_latency_buckets: Counter[float] = field(default_factory=Counter)

    def observe(
        self,
        *,
        status_code: int,
        latency_ms: float,
        verdict: str | None = None,
        label_verdict: str | None = None,
        label_distribution: dict[str, float] | None = None,
        stage_latencies_ms: dict[str, float] | None = None,
        llm_provider: str | None = None,
        error_type: str | None = None,
    ) -> None:
        with self.lock:
            self.request_total += 1
            self.request_status_total[str(status_code)] += 1
            self.total_latency_ms_sum += latency_ms
            self.total_latency_ms_count += 1
            self.total_latency_ms_max = max(self.total_latency_ms_max, latency_ms)

            for bucket in _PROMETHEUS_BUCKETS_MS:
                if latency_ms <= bucket:
                    self.total_latency_buckets[bucket] += 1
                    break
            else:
                self.total_latency_buckets[float("inf")] += 1

            if verdict:
                self.verdict_total[verdict] += 1
            if label_verdict:
                self.label_total[label_verdict] += 1
            if llm_provider:
                self.llm_provider_total[llm_provider] += 1
            if error_type:
                self.error_total[error_type] += 1

            if label_distribution:
                for label, value in _sanitize_distribution(label_distribution).items():
                    self.label_distribution_total[label] += value

            if stage_latencies_ms:
                for stage, stage_latency in stage_latencies_ms.items():
                    self.stage_latency_ms_sum[stage] += stage_latency
                    self.stage_latency_ms_count[stage] += 1
                    self.stage_latency_ms_max[stage] = max(
                        self.stage_latency_ms_max.get(stage, 0.0),
                        stage_latency,
                    )

    def to_prometheus(self, model_version: str | None = None) -> str:
        with self.lock:
            status_lines = [
                f'sicurre_inference_request_status_total{{status_code="{status_code}"}} {count}'
                for status_code, count in sorted(self.request_status_total.items())
            ]

            cumulative = 0
            latency_bucket_lines: list[str] = []
            for bucket in _PROMETHEUS_BUCKETS_MS:
                cumulative += self.total_latency_buckets.get(bucket, 0)
                latency_bucket_lines.append(
                    f'sicurre_inference_request_latency_ms_bucket{{le="{bucket}"}} {cumulative}'
                )
            cumulative += self.total_latency_buckets.get(float("inf"), 0)

            verdict_lines = [
                f'sicurre_inference_verdict_total{{verdict="{verdict}"}} {count}'
                for verdict, count in sorted(self.verdict_total.items())
            ]
            label_lines = [
                f'sicurre_inference_label_total{{label="{label}"}} {count}'
                for label, count in sorted(self.label_total.items())
            ]
            label_distribution_lines = [
                f'sicurre_inference_label_distribution_total{{label="{label}"}} {round(total, 6)}'
                for label, total in sorted(self.label_distribution_total.items())
            ]
            stage_latency_sum_lines = [
                f'sicurre_inference_stage_latency_ms_sum{{stage="{stage}"}} {round(total, 6)}'
                for stage, total in sorted(self.stage_latency_ms_sum.items())
            ]
            stage_latency_count_lines = [
                f'sicurre_inference_stage_latency_ms_count{{stage="{stage}"}} {count}'
                for stage, count in sorted(self.stage_latency_ms_count.items())
            ]
            llm_provider_lines = [
                f'sicurre_inference_llm_provider_total{{provider="{provider}"}} {count}'
                for provider, count in sorted(self.llm_provider_total.items())
            ]
            error_lines = [
                f'sicurre_inference_error_total{{error_type="{error_type}"}} {count}'
                for error_type, count in sorted(self.error_total.items())
            ]

            model_version = model_version or _env_text("MODEL_SHA", _env_text("ONNX_MODEL_SHA", "unknown"))

            lines = [
                '# HELP sicurre_inference_requests_total Total classify requests processed.',
                '# TYPE sicurre_inference_requests_total counter',
                f'sicurre_inference_requests_total {self.request_total}',
                '# HELP sicurre_inference_request_status_total Requests grouped by HTTP status.',
                '# TYPE sicurre_inference_request_status_total counter',
                *status_lines,
                '# HELP sicurre_inference_request_latency_ms Request latency in milliseconds.',
                '# TYPE sicurre_inference_request_latency_ms histogram',
                *latency_bucket_lines,
                f'sicurre_inference_request_latency_ms_bucket{{le="+Inf"}} {cumulative}',
                f'sicurre_inference_request_latency_ms_sum {round(self.total_latency_ms_sum, 6)}',
                f'sicurre_inference_request_latency_ms_count {self.total_latency_ms_count}',
                f'sicurre_inference_request_latency_ms_max {round(self.total_latency_ms_max, 6)}',
                '# HELP sicurre_inference_verdict_total Verdict counts returned by the service.',
                '# TYPE sicurre_inference_verdict_total counter',
                *verdict_lines,
                '# HELP sicurre_inference_label_total Label verdict counts returned by the service.',
                '# TYPE sicurre_inference_label_total counter',
                *label_lines,
                '# HELP sicurre_inference_label_distribution_total Sum of label distribution probabilities across requests.',
                '# TYPE sicurre_inference_label_distribution_total counter',
                *label_distribution_lines,
                '# HELP sicurre_inference_stage_latency_ms_sum Stage latency in milliseconds.',
                '# TYPE sicurre_inference_stage_latency_ms_sum counter',
                *stage_latency_sum_lines,
                '# HELP sicurre_inference_stage_latency_ms_count Stage latency observations.',
                '# TYPE sicurre_inference_stage_latency_ms_count counter',
                *stage_latency_count_lines,
                '# HELP sicurre_inference_llm_provider_total LLM provider usage counts.',
                '# TYPE sicurre_inference_llm_provider_total counter',
                *llm_provider_lines,
                '# HELP sicurre_inference_error_total Runtime error counts.',
                '# TYPE sicurre_inference_error_total counter',
                *error_lines,
                '# HELP sicurre_inference_model_info Model identity exposed as a gauge.',
                '# TYPE sicurre_inference_model_info gauge',
                f'sicurre_inference_model_info{{version="{model_version}"}} 1',
            ]

            return "\n".join(lines) + "\n"


runtime_telemetry = RuntimeTelemetry()


def emit_classify_request_log(
    *,
    request_id: str,
    status_code: int,
    latency_ms: float,
    verdict: str | None = None,
    label_verdict: str | None = None,
    label_distribution: dict[str, float] | None = None,
    stage_latencies_ms: dict[str, float] | None = None,
    llm_provider: str | None = None,
    model_version: str | None = None,
    error_type: str | None = None,
    error_detail: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "event": "classify_request",
        "request_id": request_id,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 3),
        "model_version": model_version or _env_text("MODEL_SHA", _env_text("ONNX_MODEL_SHA", "unknown")),
    }
    if verdict is not None:
        payload["verdict"] = verdict
    if label_verdict is not None:
        payload["label_verdict"] = label_verdict
    if label_distribution is not None:
        payload["label_distribution"] = _sanitize_distribution(label_distribution)
    if stage_latencies_ms is not None:
        payload["stage_latencies_ms"] = {
            stage: round(value, 3) for stage, value in stage_latencies_ms.items()
        }
    if llm_provider is not None:
        payload["llm_provider"] = llm_provider or None
    if error_type is not None:
        payload["error_type"] = error_type
    if error_detail is not None:
        payload["error_detail"] = error_detail

    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def observe_classify_request(
    *,
    status_code: int,
    latency_ms: float,
    verdict: str | None = None,
    label_verdict: str | None = None,
    label_distribution: dict[str, float] | None = None,
    stage_latencies_ms: dict[str, float] | None = None,
    llm_provider: str | None = None,
    model_version: str | None = None,
    error_type: str | None = None,
) -> None:
    runtime_telemetry.observe(
        status_code=status_code,
        latency_ms=latency_ms,
        verdict=verdict,
        label_verdict=label_verdict,
        label_distribution=label_distribution,
        stage_latencies_ms=stage_latencies_ms,
        llm_provider=llm_provider,
        error_type=error_type,
    )
