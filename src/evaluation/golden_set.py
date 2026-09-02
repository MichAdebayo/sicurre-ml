from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
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
    id: str
    subject: str
    sender: str
    text: str
    expected_label: str
    language: str
    scenario: str
    difficulty: str
    reviewer_rationale: str
    reviewed_by: str
    reviewed_at: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> GoldenSample:
        expected = {
            "id",
            "subject",
            "sender",
            "text",
            "expected_label",
            "language",
            "scenario",
            "difficulty",
            "reviewer_rationale",
            "reviewed_by",
            "reviewed_at",
        }
        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)
            raise ValueError(f"Invalid golden sample fields; missing={missing}, extra={extra}")
        if value["expected_label"] not in LABEL2ID:
            raise ValueError(
                f"Unsupported golden sample label: {value['expected_label']!r}"
            )
        for field_name in expected:
            if not isinstance(value[field_name], str):
                raise ValueError(f"Golden sample field {field_name!r} must be a string")
        required_non_empty = expected - {"subject", "sender", "text"}
        for field_name in required_non_empty:
            if not value[field_name].strip():
                raise ValueError(f"Golden sample field {field_name!r} must be non-empty")
        if not any(value[field].strip() for field in ("subject", "sender", "text")):
            raise ValueError("Golden sample must contain subject, sender, or text")
        bounds = {"subject": 500, "sender": 200, "text": 5500}
        for field_name, maximum in bounds.items():
            if len(value[field_name]) > maximum:
                raise ValueError(
                    f"Golden sample field {field_name!r} exceeds {maximum} characters"
                )
        if value["language"] not in {"fr", "en"}:
            raise ValueError(f"Unsupported golden sample language: {value['language']!r}")
        try:
            reviewed_at = datetime.fromisoformat(
                value["reviewed_at"].replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError("Golden sample reviewed_at must be ISO-8601") from exc
        if reviewed_at.tzinfo is None:
            raise ValueError("Golden sample reviewed_at must include a timezone")
        return cls(**{key: value[key].strip() for key in expected})

    @property
    def model_text(self) -> str:
        """Build the classifier input exactly as production does.

        This used to compose its own framing -- "Subject:", "From:", in English --
        which appeared neither in the training corpus nor in the serving path. The
        gate was therefore scoring candidates under conditions no deployed model
        ever meets, and the gap was not small: on this same 60-email set the
        production model scores 0.68 phishing recall in one framing and 0.84 in
        another. A promotion decision resting on that is measuring the wrapper as
        much as the model.

        Delegating to canonicalize_email removes the possibility of drift: the gate
        and the scan path now derive their input from one function, so changing how
        emails are presented to the model cannot silently change what promotion
        means.
        """
        from src.inference.input_normalizer import canonicalize_email

        return canonicalize_email(
            self.text, subject=self.subject, sender=self.sender
        ).model_text


@dataclass(frozen=True, slots=True)
class GoldenEvaluationReport:
    sample_count: int
    weighted_f1: float
    phishing_recall: float
    legitimate_false_positive_count: int
    legitimate_false_positive_rate: float
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
            "legitimate_false_positive_count": self.legitimate_false_positive_count,
            "legitimate_false_positive_rate": self.legitimate_false_positive_rate,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass(frozen=True, slots=True)
class GoldenSetRelease:
    """One published, human-approved evaluation set.

    Registered explicitly rather than discovered from storage. The gate decides
    what reaches production, so the set it scores against has to be immutable
    and reviewed: listing a bucket and trusting whatever is newest would let
    anyone with write access to that bucket change the bar.

    Publishing a new set is therefore one reviewed line here, and the evaluator
    then picks it up automatically - which is the part that was missing.
    `golden-20260816-v3` sat in R2 unused from 16 August while the gate went on
    scoring against a 60-sample set from 19 July.
    """

    version: str
    object_key: str
    sha256: str
    schema_version: str
    sample_count: int


#: Newest last. `latest_golden_set()` picks the final entry.
GOLDEN_SET_RELEASES: tuple[GoldenSetRelease, ...] = (
    GoldenSetRelease(
        version="golden-20260719-v1",
        object_key="golden.jsonl",
        sha256="bc329213cacddab409a63deb9d663e593351b6e740a45cdada4c201e3beea346",
        schema_version="1",
        sample_count=60,
    ),
    GoldenSetRelease(
        version="golden-20260816-v3",
        object_key="evaluation_sets/golden-20260816-v3/golden.jsonl",
        sha256="6d15f2141cd69d98c9b4ee9b47d505c8aae8505d900fa77705fe0c57b13fb632",
        schema_version="2",
        sample_count=95,
    ),
)


def latest_golden_set() -> GoldenSetRelease:
    """The most recent published evaluation set.

    Evaluation must track the best available measure of real behaviour. A gate
    pinned to an old set silently stops improving as the set grows, and the
    comparison it reports gets weaker over time rather than stronger.
    """
    if not GOLDEN_SET_RELEASES:
        raise ValueError("No golden set releases are registered")
    return GOLDEN_SET_RELEASES[-1]


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
            if sample.id in seen_ids:
                raise ValueError(f"Duplicate golden sample ID: {sample.id}")
            seen_ids.add(sample.id)
            samples.append(sample)

    if not samples:
        raise ValueError("Golden set is empty")
    if reference.schema_version == "2":
        # Schema 2: the set is expected to grow, so its size is not pinned.
        # What must hold is that it still measures the thing it exists to
        # measure - all three classes present, French only. Pinning exact counts
        # here would mean every published set needed a code change to be
        # loadable, which is how v3 ended up sitting in storage unused.
        present = {sample.expected_label for sample in samples}
        missing = {"phishing", "spam", "legitimate"} - present
        if missing:
            raise ValueError(f"Golden set is missing classes: {sorted(missing)}")
        if any(sample.language != "fr" for sample in samples):
            raise ValueError("Golden set must contain French examples only")
        return samples

    if reference.schema_version == "1":
        expected_counts = {"phishing": 25, "legitimate": 25, "spam": 10}
        actual_counts = {
            label: sum(sample.expected_label == label for sample in samples)
            for label in expected_counts
        }
        if actual_counts != expected_counts:
            raise ValueError(
                f"Golden-set v1 class counts invalid: expected {expected_counts}, "
                f"got {actual_counts}"
            )
        if any(sample.language != "fr" for sample in samples):
            raise ValueError("Golden-set v1 must contain French examples only")
    return samples


def evaluate_golden_set(
    samples: list[GoldenSample],
    predict: Callable[[str], str],
) -> GoldenEvaluationReport:
    """Evaluate a model without retaining sample text in the report."""
    if not samples:
        raise ValueError("Golden set is empty")
    expected = [sample.expected_label for sample in samples]
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
    legitimate_count = sum(label == "legitimate" for label in expected)
    legitimate_false_positives = sum(
        actual == "legitimate" and guess == "phishing"
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
        legitimate_false_positive_count=legitimate_false_positives,
        legitimate_false_positive_rate=(
            legitimate_false_positives / legitimate_count if legitimate_count else 0.0
        ),
        p50_latency_ms=float(np.percentile(latencies_ms, 50)),
        p95_latency_ms=float(np.percentile(latencies_ms, 95)),
        p99_latency_ms=float(np.percentile(latencies_ms, 99)),
        per_class=per_class,
        confusion_matrix=confusion_matrix(expected, predicted, labels=labels).tolist(),
    )
