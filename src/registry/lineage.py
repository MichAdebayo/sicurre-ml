from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

PromotionState = Literal["candidate", "approved", "rejected", "inconclusive"]


@dataclass(frozen=True, slots=True)
class DatasetLineage:
    dataset_id: str
    version: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ModelLineage:
    semantic_version: str
    mlflow_run_id: str
    mlflow_model_name: str
    mlflow_model_version: str
    huggingface_repo: str
    huggingface_revision: str


@dataclass(frozen=True, slots=True)
class TrainingManifest:
    schema_version: int
    state: PromotionState
    created_at: str
    source_revision: str
    github_run_id: str
    training_dataset: DatasetLineage
    model: ModelLineage
    golden_set: DatasetLineage | None = None
    gate: dict[str, Any] | None = None
    approver: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def new_training_manifest(
    *,
    state: PromotionState,
    source_revision: str,
    github_run_id: str,
    training_dataset: DatasetLineage,
    model: ModelLineage,
    golden_set: DatasetLineage | None = None,
    gate: dict[str, Any] | None = None,
    approver: str | None = None,
) -> TrainingManifest:
    return TrainingManifest(
        schema_version=1,
        state=state,
        created_at=datetime.now(UTC).isoformat(),
        source_revision=source_revision,
        github_run_id=github_run_id,
        training_dataset=training_dataset,
        model=model,
        golden_set=golden_set,
        gate=gate,
        approver=approver,
    )


def mlflow_lineage_tags(manifest: TrainingManifest) -> dict[str, str]:
    tags = {
        "sicurre.promotion.state": manifest.state,
        "sicurre.source.revision": manifest.source_revision,
        "sicurre.github.run_id": manifest.github_run_id,
        "sicurre.dataset.id": manifest.training_dataset.dataset_id,
        "sicurre.dataset.version": manifest.training_dataset.version,
        "sicurre.dataset.sha256": manifest.training_dataset.sha256,
        "sicurre.model.semantic_version": manifest.model.semantic_version,
        "sicurre.model.hf_revision": manifest.model.huggingface_revision,
    }
    if manifest.golden_set is not None:
        tags.update(
            {
                "sicurre.golden_set.id": manifest.golden_set.dataset_id,
                "sicurre.golden_set.version": manifest.golden_set.version,
                "sicurre.golden_set.sha256": manifest.golden_set.sha256,
            }
        )
    return tags
