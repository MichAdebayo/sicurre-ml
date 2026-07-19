from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.golden_set import (
    GoldenSample,
    GoldenSetReference,
    evaluate_golden_set,
    file_sha256,
    load_approved_golden_set,
)


def _sample(sample_id: str = "gold-001") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "subject": "Account review",
        "sender": "security@example.invalid",
        "text": "Review the notice at hxxps://example[.]invalid.",
        "label": "phishing",
        "language": "en",
        "scenario": "credential theft",
        "difficulty": "hard",
        "reviewer_rationale": "Urgency and a defanged credential lure.",
    }


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _reference(path: Path) -> GoldenSetReference:
    return GoldenSetReference(
        dataset_id="sicurre-golden",
        version="golden-20260719-001",
        sha256=file_sha256(path),
        schema_version="1",
        provenance="synthetic_provisional",
        review_status="approved",
    )


def test_loads_checksum_verified_approved_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    _write_jsonl(path, [_sample()])

    samples = load_approved_golden_set(path, _reference(path))

    assert samples[0].sample_id == "gold-001"
    assert "Subject: Account review" in samples[0].model_text


def test_rejects_checksum_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    _write_jsonl(path, [_sample()])
    reference = _reference(path)
    path.write_text(path.read_text() + "\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_approved_golden_set(path, reference)


def test_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    _write_jsonl(path, [_sample(), _sample()])

    with pytest.raises(ValueError, match="Duplicate golden sample ID"):
        load_approved_golden_set(path, _reference(path))


def test_evaluation_report_contains_metrics_but_no_sample_content() -> None:
    sample = _sample()
    report = evaluate_golden_set(
        [GoldenSample.from_mapping(sample)],
        lambda _: "phishing",
    )

    assert report.phishing_recall == 1.0
    assert "Account review" not in json.dumps(report.to_dict())
