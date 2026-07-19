from __future__ import annotations

import json
from pathlib import Path

from src.registry.lineage import (
    DatasetLineage,
    ModelLineage,
    mlflow_lineage_tags,
    new_training_manifest,
)


def test_manifest_is_machine_readable_and_tags_are_bounded(tmp_path: Path) -> None:
    manifest = new_training_manifest(
        state="candidate",
        source_revision="abc123",
        github_run_id="42",
        training_dataset=DatasetLineage("sicurre-data", "base-1", "a" * 64),
        model=ModelLineage(
            semantic_version="2.0.0",
            mlflow_run_id="run-1",
            mlflow_model_name="main.sicurre.phishing-detector",
            mlflow_model_version="17",
            huggingface_repo="owner/model",
            huggingface_revision="b" * 40,
        ),
    )

    path = manifest.write_json(tmp_path / "manifest.json")
    payload = json.loads(path.read_text())
    tags = mlflow_lineage_tags(manifest)

    assert payload["state"] == "candidate"
    assert tags["sicurre.dataset.version"] == "base-1"
    assert tags["sicurre.model.hf_revision"] == "b" * 40
