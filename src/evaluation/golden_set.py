from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

from src.config.training_config import LABEL2ID

ReviewStatus = Literal["approved"]
Provenance = Literal["synthetic_provisional", "real_world_reviewed"]


@dataclass(frozen=True, slots=True)
class GoldenSetReference:
    dataset_id: str
    version: str
    sha256: str
    schema_version: str
    provenance: Provenance
    review_status: ReviewStatus


@dataclass(frozen=True, slots=True)
class GoldenSample:
    sample_id: str
    subject: str
    sender: str
    text: str
    label: str
    language: str
    scenario: str
    difficulty: str
    reviewer_rationale: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> GoldenSample:
        expected = {
            "sample_id",
            "subject",
            "sender",
            "text",
            "label",
            "language",
            "scenario",
            "difficulty",
            "reviewer_rationale",
        }
        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)
            raise ValueError(f"Invalid golden sample fields; missing={missing}, extra={extra}")
        if value["label"] not in LABEL2ID:
            raise ValueError(f"Unsupported golden sample label: {value['label']!r}")
        for field_name in expected:
            if not isinstance(value[field_name], str) or not value[field_name].strip():
                raise ValueError(f"Golden sample field {field_name!r} must be non-empty")
        return cls(**{key: value[key].strip() for key in expected})

    @property
    def model_text(self) -> str:
        return "\n".join(
            part
            for part in (
                f"Subject: {self.subject}",
                f"From: {self.sender}",
                self.text,
            )
            if part
        )


@dataclass(frozen=True, slots=True)
class GoldenEvaluationReport:
    sample_count: int
    weighted_f1: float
    phishing_recall: float
    phishing_false_positive_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "weighted_f1": self.weighted_f1,
            "phishing_recall": self.phishing_recall,
            "phishing_false_positive_rate": self.phishing_false_positive_rate,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_approved_golden_set(
    path: Path,
    reference: GoldenSetReference,
) -> list[GoldenSample]:
    """Load an approved immutable JSONL evaluation set.

    The caller must obtain ``path`` through the evaluation-only Sicurre
    contract. This loader deliberately has no training-dataset discovery or
    download behavior.
    """
    if reference.review_status != "approved":
        raise ValueError("Golden set must be human-reviewed and approved")
    actual_checksum = file_sha256(path)
    if actual_checksum != reference.sha256.lower():
        raise ValueError(
            f"Golden set checksum mismatch: expected {reference.sha256}, got {actual_checksum}"
        )

    samples: list[GoldenSample] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on golden-set line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Golden-set line {line_number} must be an object")
            sample = GoldenSample.from_mapping(value)
            if sample.sample_id in seen_ids:
                raise ValueError(f"Duplicate golden sample ID: {sample.sample_id}")
            seen_ids.add(sample.sample_id)
            samples.append(sample)

    if not samples:
        raise ValueError("Golden set is empty")
    return samples


def evaluate_golden_set(
    samples: list[GoldenSample],
    predict: Callable[[str], str],
) -> GoldenEvaluationReport:
    """Evaluate a model without retaining sample text in the report."""
    if not samples:
        raise ValueError("Golden set is empty")
    expected = [sample.label for sample in samples]
    predicted: list[str] = []
    latencies_ms: list[float] = []
    for sample in samples:
        started = time.perf_counter()
        label = predict(sample.model_text)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        if label not in LABEL2ID:
            raise ValueError(f"Predictor returned unsupported label: {label!r}")
        predicted.append(label)

    labels = list(LABEL2ID)
    report = classification_report(
        expected,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    non_phishing = sum(label != "phishing" for label in expected)
    phishing_false_positives = sum(
        actual != "phishing" and guess == "phishing"
        for actual, guess in zip(expected, predicted, strict=True)
    )
    per_class = {
        label: {
            metric: float(report[label][metric])
            for metric in ("precision", "recall", "f1-score", "support")
        }
        for label in labels
    }
    return GoldenEvaluationReport(
        sample_count=len(samples),
        weighted_f1=float(f1_score(expected, predicted, average="weighted")),
        phishing_recall=float(
            recall_score(
                expected,
                predicted,
                labels=["phishing"],
                average="macro",
                zero_division=0,
            )
        ),
        phishing_false_positive_rate=(
            phishing_false_positives / non_phishing if non_phishing else 0.0
        ),
        p50_latency_ms=float(np.percentile(latencies_ms, 50)),
        p95_latency_ms=float(np.percentile(latencies_ms, 95)),
        p99_latency_ms=float(np.percentile(latencies_ms, 99)),
        per_class=per_class,
        confusion_matrix=confusion_matrix(expected, predicted, labels=labels).tolist(),
    )
