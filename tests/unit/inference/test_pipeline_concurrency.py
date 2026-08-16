"""Concurrency and failover-budget contracts for the inference pipeline.

A single shared two-worker pool previously serialised the service: each request
submits both an ONNX task and an LLM task, so one in-flight request saturated
the pool and a second concurrent scan could not begin local inference until the
first request's LLM call returned.
"""

from __future__ import annotations

import importlib

import pytest

from src.inference import llm_classifier, pipeline


def test_semantic_stages_do_not_share_a_worker_pool() -> None:
    """CPU-bound and I/O-bound stages must not contend for the same workers."""
    assert pipeline._ONNX_EXECUTOR is not pipeline._LLM_EXECUTOR


def test_llm_pool_admits_more_concurrent_requests_than_the_cpu_pool() -> None:
    """The LLM stage waits on a socket, so it is not bounded by core count."""
    assert pipeline._LLM_EXECUTOR._max_workers > pipeline._ONNX_EXECUTOR._max_workers
    # One in-flight request must no longer be able to saturate either pool.
    assert pipeline._LLM_EXECUTOR._max_workers >= 2


def test_pool_sizes_are_configurable_and_never_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONNX_POOL_WORKERS", "4")
    assert pipeline._pool_size("ONNX_POOL_WORKERS", 1) == 4

    # A pool of zero workers would deadlock every request.
    monkeypatch.setenv("ONNX_POOL_WORKERS", "0")
    assert pipeline._pool_size("ONNX_POOL_WORKERS", 2) == 1

    monkeypatch.setenv("ONNX_POOL_WORKERS", "not-a-number")
    assert pipeline._pool_size("ONNX_POOL_WORKERS", 3) == 3


def test_shutdown_releases_both_pools(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.reload(pipeline)
    closed: list[str] = []
    monkeypatch.setattr(
        module._ONNX_EXECUTOR, "shutdown", lambda **_: closed.append("onnx")
    )
    monkeypatch.setattr(
        module._LLM_EXECUTOR, "shutdown", lambda **_: closed.append("llm")
    )

    module.close_pipeline_resources()

    assert sorted(closed) == ["llm", "onnx"]


def test_primary_provider_timeout_fails_over_before_the_delivery_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured median provider latency is ~1.1s.

    A 4.5s ceiling never helped a healthy call; it only delayed failover on a
    stalled one, which is what inflated p95/p99. The primary timeout must stay
    comfortably above the median and well below the total chain budget, so a
    stalled provider can be abandoned and a second one still tried.
    """
    monkeypatch.delenv("LLM_PROVIDER_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LLM_TOTAL_TIMEOUT_SECONDS", raising=False)

    per_provider = llm_classifier._env_float("LLM_PROVIDER_TIMEOUT_SECONDS", 2.5)
    total_budget = llm_classifier._env_float("LLM_TOTAL_TIMEOUT_SECONDS", 7.5)

    assert per_provider == 2.5
    assert per_provider > 1.5, "must not cut off a normal ~1.1s response"
    assert per_provider * 2 <= total_budget, "two providers must fit in the budget"
