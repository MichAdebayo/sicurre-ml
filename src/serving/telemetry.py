from __future__ import annotations

import json
import os
import resource
import sys
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
    auth_failure_total: int = 0
    rate_limit_total: int = 0
    model_ready: int = 0
    total_latency_ms_sum: float = 0.0
    total_latency_ms_count: int = 0
    total_latency_ms_max: float = 0.0
    stage_latency_ms_sum: dict[str, float] = field(default_factory=dict)
    stage_latency_ms_count: Counter[str] = field(default_factory=Counter)
    stage_latency_ms_max: dict[str, float] = field(default_factory=dict)
    label_distribution_total: dict[str, float] = field(default_factory=dict)
    total_latency_buckets: Counter[float] = field(default_factory=Counter)
    mode_latency_buckets: Counter[tuple[str, float]] = field(default_factory=Counter)
    mode_latency_sum: dict[str, float] = field(default_factory=dict)
    mode_latency_count: Counter[str] = field(default_factory=Counter)

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
        mode: str = "unknown",
    ) -> None:
        with self.lock:
            self.request_total += 1
            self.request_status_total[str(status_code)] += 1
            self.total_latency_ms_sum += latency_ms
            self.total_latency_ms_count += 1
            self.total_latency_ms_max = max(self.total_latency_ms_max, latency_ms)
            bounded_mode = mode if mode in {"local", "llm"} else "unknown"
            self.mode_latency_sum[bounded_mode] = (
                self.mode_latency_sum.get(bounded_mode, 0.0) + latency_ms
            )
            self.mode_latency_count[bounded_mode] += 1

            for bucket in _PROMETHEUS_BUCKETS_MS:
                if latency_ms <= bucket:
                    self.total_latency_buckets[bucket] += 1
                    self.mode_latency_buckets[(bounded_mode, bucket)] += 1
                    break
            else:
                self.total_latency_buckets[float("inf")] += 1
                self.mode_latency_buckets[(bounded_mode, float("inf"))] += 1

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
                    self.label_distribution_total[label] = (
                        self.label_distribution_total.get(label, 0.0) + value
                    )

            if stage_latencies_ms:
                for stage, stage_latency in stage_latencies_ms.items():
                    self.stage_latency_ms_sum[stage] = (
                        self.stage_latency_ms_sum.get(stage, 0.0) + stage_latency
                    )
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

            mode_latency_lines: list[str] = []
            for mode in sorted(self.mode_latency_count):
                mode_cumulative = 0
                for bucket in _PROMETHEUS_BUCKETS_MS:
                    mode_cumulative += self.mode_latency_buckets.get((mode, bucket), 0)
                    mode_latency_lines.append(
                        'sicurre_inference_mode_request_latency_ms_bucket'
                        f'{{mode="{mode}",le="{bucket}"}} {mode_cumulative}'
                    )
                mode_cumulative += self.mode_latency_buckets.get((mode, float("inf")), 0)
                mode_latency_lines.extend(
                    [
                        'sicurre_inference_mode_request_latency_ms_bucket'
                        f'{{mode="{mode}",le="+Inf"}} {mode_cumulative}',
                        'sicurre_inference_mode_request_latency_ms_sum'
                        f'{{mode="{mode}"}} {round(self.mode_latency_sum[mode], 6)}',
                        'sicurre_inference_mode_request_latency_ms_count'
                        f'{{mode="{mode}"}} {self.mode_latency_count[mode]}',
                    ]
                )

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

            model_version = model_version or _env_text(
                "MODEL_SHA", _env_text("ONNX_MODEL_SHA", "unknown")
            )

            lines = [
                '# HELP sicurre_service_up Process liveness reported by the metrics endpoint.',
                '# TYPE sicurre_service_up gauge',
                'sicurre_service_up 1',
                '# HELP sicurre_model_ready Last observed model readiness state.',
                '# TYPE sicurre_model_ready gauge',
                f'sicurre_model_ready {self.model_ready}',
                '# HELP sicurre_auth_failure_total Rejected bearer authentication attempts.',
                '# TYPE sicurre_auth_failure_total counter',
                f'sicurre_auth_failure_total {self.auth_failure_total}',
                '# HELP sicurre_rate_limit_total Requests rejected by application rate limiting.',
                '# TYPE sicurre_rate_limit_total counter',
                f'sicurre_rate_limit_total {self.rate_limit_total}',
                '# HELP sicurre_process_resident_memory_bytes Process resident memory.',
                '# TYPE sicurre_process_resident_memory_bytes gauge',
                f'sicurre_process_resident_memory_bytes {_resident_memory_bytes()}',
                '# HELP sicurre_process_cpu_seconds_total Process CPU time.',
                '# TYPE sicurre_process_cpu_seconds_total counter',
                f'sicurre_process_cpu_seconds_total {_process_cpu_seconds()}',
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
                '# HELP sicurre_inference_mode_request_latency_ms Latency by bounded mode.',
                '# TYPE sicurre_inference_mode_request_latency_ms histogram',
                *mode_latency_lines,
                '# HELP sicurre_inference_verdict_total Verdict counts returned by the service.',
                '# TYPE sicurre_inference_verdict_total counter',
                *verdict_lines,
                '# HELP sicurre_inference_label_total Label verdict counts.',
                '# TYPE sicurre_inference_label_total counter',
                *label_lines,
                '# HELP sicurre_inference_label_distribution_total Summed label probabilities.',
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

    def observe_auth_failure(self) -> None:
        with self.lock:
            self.auth_failure_total += 1

    def observe_rate_limit(self) -> None:
        with self.lock:
            self.rate_limit_total += 1

    def set_model_ready(self, ready: bool) -> None:
        with self.lock:
            self.model_ready = int(ready)


runtime_telemetry = RuntimeTelemetry()


def _resident_memory_bytes() -> int:
    try:
        with open("/proc/self/statm", encoding="utf-8") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, IndexError, OSError, ValueError):
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage * (1 if sys.platform == "darwin" else 1024))


def _process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return round(usage.ru_utime + usage.ru_stime, 6)


def emit_classify_request_log(
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
    mode: str = "unknown",
) -> None:
    payload: dict[str, Any] = {
        "event": "classify_request",
        "status_code": status_code,
        "latency_ms": round(latency_ms, 3),
        "model_version": model_version
        or _env_text("MODEL_SHA", _env_text("ONNX_MODEL_SHA", "unknown")),
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
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def emit_operational_log(event: str, *, category: str, status_code: int) -> None:
    print(
        json.dumps(
            {"event": event, "category": category, "status_code": status_code},
            sort_keys=True,
        ),
        flush=True,
    )


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
    mode: str = "unknown",
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
        mode=mode,
    )
