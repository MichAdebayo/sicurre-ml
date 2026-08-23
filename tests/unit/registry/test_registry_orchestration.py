from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.config.training_config import RuntimeState
from src.registry import (
    publish_candidate_to_hub,
    setup_mlflow,
    stage_candidate,
    tag_registered_model_lineage,
)


class FakeMlflowClient:
    def __init__(self) -> None:
        self.aliases: list[tuple[str, str, str]] = []
        self.run_tags: list[tuple[str, str, str]] = []
        self.version_tags: list[tuple[str, str, str, str]] = []

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases.append((name, alias, version))

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.run_tags.append((run_id, key, value))

    def set_model_version_tag(
        self, name: str, version: str, key: str, value: str
    ) -> None:
        self.version_tags.append((name, version, key, value))


def _fake_mlflow(client: FakeMlflowClient) -> ModuleType:
    module = ModuleType("mlflow")
    setattr(module, "tracking_uri", "")
    setattr(module, "experiment", "")
    setattr(module, "registry_uri", "")
    setattr(module, "set_tracking_uri", lambda value: setattr(module, "tracking_uri", value))
    setattr(module, "set_experiment", lambda value: setattr(module, "experiment", value))
    setattr(module, "set_registry_uri", lambda value: setattr(module, "registry_uri", value))
    setattr(module, "MlflowClient", lambda: client)
    return module


def _runtime_state(tmp_path: Path, *, databricks: bool) -> RuntimeState:
    return RuntimeState(
        runtime_env="local",
        device="cpu",
        use_tpu=False,
        run_date="2026-08-23",
        data_dir=tmp_path,
        output_dir=tmp_path,
        secrets={},
        hf_token=None,
        databricks_host="https://workspace" if databricks else None,
        databricks_token="token" if databricks else None,
        databricks_email="owner@example.com" if databricks else None,
        mlflow_experiment_name="sicurre-training",
    )


def test_setup_mlflow_selects_local_or_databricks(monkeypatch, tmp_path: Path) -> None:
    client = FakeMlflowClient()
    fake_mlflow = _fake_mlflow(client)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    local_path = setup_mlflow(_runtime_state(tmp_path, databricks=False))
    assert local_path == "sicurre-training"
    assert getattr(fake_mlflow, "tracking_uri") == "file:./mlruns"

    remote_path = setup_mlflow(_runtime_state(tmp_path, databricks=True))
    assert remote_path == "/Users/owner@example.com/sicurre-training"
    assert getattr(fake_mlflow, "tracking_uri") == "databricks"
    assert os.environ["DATABRICKS_HOST"] == "https://workspace"


def test_stage_candidate_skips_without_databricks(monkeypatch, capsys) -> None:
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    stage_candidate("model", "3")

    assert "skipping candidate alias" in capsys.readouterr().out


def test_stage_candidate_records_alias_and_lineage(monkeypatch) -> None:
    client = FakeMlflowClient()
    fake_mlflow = _fake_mlflow(client)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token")

    stage_candidate("model", "3", semantic_version="1.0.3", run_id="run-3")

    assert client.aliases == [("model", "candidate", "3")]
    assert ("run-3", "sicurre.model.stage", "candidate") in client.run_tags
    assert ("run-3", "mlflow.runName", "model-1.0.3-candidate") in client.run_tags


def test_registered_lineage_is_mirrored_to_run_and_version(monkeypatch) -> None:
    client = FakeMlflowClient()
    monkeypatch.setitem(sys.modules, "mlflow", _fake_mlflow(client))

    tag_registered_model_lineage(
        "model",
        "4",
        "run-4",
        {"sicurre.dataset.version": "dataset-v4"},
    )

    assert client.run_tags == [("run-4", "sicurre.dataset.version", "dataset-v4")]
    assert client.version_tags == [
        ("model", "4", "sicurre_dataset_version", "dataset-v4")
    ]


def test_publish_candidate_uploads_complete_immutable_bundle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    pushes: list[tuple[str, str]] = []
    uploads: list[str] = []

    class Artifact:
        def __init__(self, kind: str) -> None:
            self.kind = kind

        def push_to_hub(self, repo_id: str, **_: object) -> None:
            pushes.append((self.kind, repo_id))

    class Factory:
        kind = "artifact"

        @classmethod
        def from_pretrained(cls, _: str) -> Artifact:
            return Artifact(cls.kind)

    class ModelFactory(Factory):
        kind = "model"

    class TokenizerFactory(Factory):
        kind = "tokenizer"

    class FakeApi:
        def upload_file(self, **kwargs: object) -> None:
            uploads.append(str(kwargs["path_in_repo"]))

        def model_info(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(sha=revision)

    (tmp_path / "model.onnx").write_bytes(b"onnx")
    artifact_dir = tmp_path / "evidence"
    artifact_dir.mkdir()
    (artifact_dir / "classification_report.txt").write_text("report")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForSequenceClassification=ModelFactory,
            AutoTokenizer=TokenizerFactory,
        ),
    )

    result = publish_candidate_to_hub(
        tmp_path,
        "owner/model",
        "token",
        {"test_f1_weighted": 0.9, "test_phishing_recall": 0.8},
        mlflow_version="4",
        artifact_dir=artifact_dir,
    )

    assert pushes == [("model", "owner/model"), ("tokenizer", "owner/model")]
    assert uploads == ["classification_report.txt", "model.onnx"]
    assert result.revision == revision


def test_publish_candidate_requires_onnx_artifact(monkeypatch, tmp_path: Path) -> None:
    class Artifact:
        def push_to_hub(self, *_: object, **__: object) -> None:
            pass

    factory = SimpleNamespace(from_pretrained=lambda _: Artifact())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForSequenceClassification=factory,
            AutoTokenizer=factory,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: SimpleNamespace()),
    )

    with pytest.raises(FileNotFoundError, match="model.onnx"):
        publish_candidate_to_hub(tmp_path, "owner/model", "token", {})
