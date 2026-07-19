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


def _sample(
    sample_id: str = "golden-001",
    label: str = "phishing",
) -> dict[str, str]:
    return {
        "id": sample_id,
        "subject": "Vérification du compte",
        "sender": "security@example.invalid",
        "text": "Consultez la notification sur hxxps://example[.]invalid.",
        "expected_label": label,
        "language": "fr",
        "scenario": "vol d'identifiants",
        "difficulty": "difficile",
        "reviewer_rationale": "Urgence associée à un lien neutralisé.",
        "reviewed_by": "owner@sicurre.example",
        "reviewed_at": "2026-07-19T10:00:00Z",
    }


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _reference(path: Path) -> GoldenSetReference:
    return GoldenSetReference(
        dataset_id="sicurre-golden",
        version="golden-20260719-001",
        sha256=file_sha256(path),
        schema_version="test",
        provenance="synthetic_provisional",
        review_status="approved",
    )


def test_loads_checksum_verified_approved_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    _write_jsonl(path, [_sample()])

    samples = load_approved_golden_set(path, _reference(path))

    assert samples[0].id == "golden-001"
    assert "Subject: Vérification du compte" in samples[0].model_text


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
    assert "Vérification du compte" not in json.dumps(report.to_dict())


def test_v1_requires_exact_counts_and_french_only(tmp_path: Path) -> None:
    rows = [
        *[_sample(f"golden-p-{i:02d}", "phishing") for i in range(25)],
        *[_sample(f"golden-l-{i:02d}", "legitimate") for i in range(25)],
        *[_sample(f"golden-s-{i:02d}", "spam") for i in range(10)],
    ]
    path = tmp_path / "golden.jsonl"
    _write_jsonl(path, rows)
    reference = _reference(path)
    reference = GoldenSetReference(
        dataset_id=reference.dataset_id,
        version=reference.version,
        sha256=reference.sha256,
        schema_version="1",
        provenance=reference.provenance,
        review_status=reference.review_status,
    )

    assert len(load_approved_golden_set(path, reference)) == 60


def test_false_positive_rate_uses_legitimate_denominator_only() -> None:
    samples = [
        GoldenSample.from_mapping(_sample("golden-l-01", "legitimate")),
        GoldenSample.from_mapping(_sample("golden-s-01", "spam")),
    ]

    report = evaluate_golden_set(samples, lambda _: "phishing")

    assert report.legitimate_false_positive_count == 1
    assert report.legitimate_false_positive_rate == 1.0
