"""The per-request log must record which stage produced which label, and the mode.

`emit_classify_request_log` accepted `stage_labels` and `mode` and wrote neither
to the payload, so the structured log could not answer the first question worth
asking about a bad verdict: did the LLM override the ONNX stage, or was it even
consulted? The Prometheus recorder builds that agreement matrix in aggregate,
but aggregate counters cannot explain one specific request.

These bind the log to the two fields so a single line is diagnosable on its own.
"""

from __future__ import annotations

import json

from src.serving.telemetry import emit_classify_request_log


def _emit_and_parse(capsys, **kwargs) -> dict:
    emit_classify_request_log(**kwargs)
    line = capsys.readouterr().out.strip().splitlines()[-1]
    return json.loads(line)


def test_stage_labels_reach_the_log(capsys) -> None:
    payload = _emit_and_parse(
        capsys,
        status_code=200,
        latency_ms=123.4,
        stage_labels={"onnx": "phishing", "llm": "legitimate"},
    )
    assert payload["stage_labels"] == {"onnx": "phishing", "llm": "legitimate"}


def test_an_llm_override_is_visible_in_a_single_line(capsys) -> None:
    """The exact case the aggregate cannot explain: onnx and llm disagree."""
    payload = _emit_and_parse(
        capsys,
        status_code=200,
        latency_ms=90.0,
        stage_labels={"onnx": "legitimate", "llm": "phishing"},
        mode="llm",
    )
    assert payload["stage_labels"]["onnx"] != payload["stage_labels"]["llm"]
    assert payload["mode"] == "llm"


def test_mode_is_always_present_and_bounded(capsys) -> None:
    """mode is recorded on every request, and only from a closed set."""
    llm = _emit_and_parse(capsys, status_code=200, latency_ms=10.0, mode="llm")
    local = _emit_and_parse(capsys, status_code=200, latency_ms=10.0, mode="local")
    assert llm["mode"] == "llm"
    assert local["mode"] == "local"


def test_an_unexpected_mode_collapses_to_unknown(capsys) -> None:
    """A stray value must not widen the log's value space.

    The recorder bounds mode to {local, llm}; the log applies the same bound so
    the two never disagree about what a mode can be.
    """
    payload = _emit_and_parse(
        capsys, status_code=200, latency_ms=10.0, mode="experimental-v3"
    )
    assert payload["mode"] == "unknown"


def test_a_503_without_a_mode_reads_unknown_not_a_crash(capsys) -> None:
    """The failure path passes no mode; it must still log a bounded value."""
    payload = _emit_and_parse(capsys, status_code=503, latency_ms=5.0)
    assert payload["mode"] == "unknown"
    assert "stage_labels" not in payload


def test_stage_labels_are_omitted_when_empty_rather_than_logged_blank(capsys) -> None:
    """No stage labels means the key is absent, not an empty object."""
    payload = _emit_and_parse(capsys, status_code=200, latency_ms=5.0, stage_labels={})
    assert "stage_labels" not in payload
