import runpy
from pathlib import Path
from typing import Any, Callable, cast

import pytest


def _script() -> dict[str, Any]:
    return runpy.run_path(str(Path(".github/scripts/promote_model.py")))


def test_promotion_inputs_require_immutable_sha_and_semver() -> None:
    namespace = _script()
    validate_sha = cast(Callable[[str, str], str], namespace["_validated_sha"])
    validate_semver = cast(Callable[[str], str], namespace["_validated_semver"])

    assert validate_sha("a" * 40, "revision") == "a" * 40
    assert validate_semver("1.0.17") == "1.0.17"
    with pytest.raises(ValueError):
        validate_sha("production", "revision")
    with pytest.raises(ValueError):
        validate_semver("20260718")


def test_protected_workflow_is_transactional_and_identity_aware() -> None:
    workflow = Path(".github/workflows/promote-model.yml").read_text(encoding="utf-8")

    assert "environment: production" in workflow
    assert "sicurre.evaluation.outcome" not in workflow
    assert "promote_model.py promote" in workflow
    assert "promote_model.py rollback" in workflow
    assert "EXPECTED_MODEL_REVISION" in workflow
    assert "EXPECTED_MODEL_VERSION" in workflow
    assert "--status active" in workflow
    assert "--status rolled_back" in workflow
    assert "steps.deploy.outcome != 'success'" in workflow
    assert "steps.active_callback.outcome != 'success'" in workflow
    assert "force-recreate app" in workflow


def test_promotion_script_reverifies_mlflow_and_hugging_face_evidence() -> None:
    script = Path(".github/scripts/promote_model.py").read_text(encoding="utf-8")

    for tag in (
        "sicurre.evaluation.outcome",
        "sicurre.candidate.run_id",
        "sicurre.candidate.mlflow_model_version",
        "sicurre.candidate.hf_revision",
        "sicurre.incumbent.hf_revision",
        "sicurre.model.semantic_version",
        "sicurre.model.stage",
    ):
        assert tag in script
    assert "get_model_version_by_alias" in script
    assert '"production"' in script
    assert '"model.onnx"' in script
    assert "_restore_registry" in script
    assert "/internal/ml/deployments" in script


class _FakeVersion:
    def __init__(self, version: str, run_id: str, description: str, tags: dict[str, str]):
        self.version = version
        self.run_id = run_id
        self.description = description
        self.tags = tags


class _FakeClient:
    """Records the registry writes a promotion makes, nothing more."""

    def __init__(self, production: _FakeVersion | None) -> None:
        self.production = production
        self.run_tags: list[tuple[str, str, str]] = []
        self.version_tags: list[tuple[str, str, str]] = []
        self.descriptions: list[tuple[str, str]] = []

    def get_model_version_by_alias(self, name: str, alias: str) -> _FakeVersion:
        if alias == "production" and self.production:
            return self.production
        raise RuntimeError("RESOURCE_DOES_NOT_EXIST: alias not found")

    def get_model_version(self, name: str, version: str) -> _FakeVersion:
        assert self.production and version == self.production.version
        return self.production

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.run_tags.append((run_id, key, value))

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self.version_tags.append((version, key, value))

    def update_model_version(self, name: str, version: str, description: str) -> None:
        self.descriptions.append((version, description))


def test_a_promoted_version_is_named_after_its_outcome() -> None:
    """The run name, the state tag and the description all say production."""
    namespace = _script()
    describe = namespace["_describe_version"]
    client = _FakeClient(None)

    describe(
        client,
        version="30",
        run_id="run-30",
        semantic_version="1.0.30",
        state="production",
        note=(
            "Production since 2026-09-02, approved by MichAdebayo, Hugging Face revision c6e4cbb7."
        ),
    )

    assert ("run-30", "mlflow.runName", "model-1.0.30-production") in client.run_tags
    assert ("run-30", "sicurre.promotion.state", "production") in client.run_tags
    assert any(
        key.endswith("promotion_state") and value == "production"
        for _, key, value in client.version_tags
    )
    assert client.descriptions == [
        (
            "30",
            "Production since 2026-09-02, approved by MichAdebayo, Hugging Face revision c6e4cbb7.",
        )
    ]


def test_a_retired_version_keeps_its_promotion_line() -> None:
    """Retirement appends to the description instead of erasing the promotion record."""
    namespace = _script()
    describe = namespace["_describe_version"]
    incumbent = _FakeVersion(
        "30", "run-30", "Production since 2026-09-02, approved by MichAdebayo.", {}
    )
    client = _FakeClient(incumbent)

    describe(
        client,
        version="30",
        run_id="run-30",
        semantic_version=None,
        state="retired",
        note="Retired 2026-09-20, replaced by 1.0.33.",
        append=True,
    )

    assert ("run-30", "mlflow.runName", "model-v30-retired") in client.run_tags
    assert client.descriptions == [
        (
            "30",
            "Production since 2026-09-02, approved by "
            "MichAdebayo.\nRetired 2026-09-20, replaced by 1.0.33.",
        )
    ]


def test_annotate_production_reads_the_alias_and_its_own_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catching up a version promoted before the naming uses the tags the promotion left."""
    namespace = _script()
    key = namespace["model_version_tag_key"]
    incumbent = _FakeVersion(
        "30",
        "run-30",
        "",
        {
            key("sicurre.model.semantic_version"): "1.0.30",
            key("sicurre.promotion.approved_by"): "MichAdebayo",
            key("sicurre.promotion.completed_at"): "2026-09-02T22:31:19Z",
            key("sicurre.model.hf_revision"): "c6e4cbb79925d3b94d85b37866b6c8ea39dc20af",
        },
    )
    client = _FakeClient(incumbent)
    namespace["annotate_production"].__globals__["_configure_mlflow"] = lambda: client

    namespace["annotate_production"](None)

    assert ("run-30", "mlflow.runName", "model-1.0.30-production") in client.run_tags
    assert client.descriptions == [
        (
            "30",
            "Production since 2026-09-02T22:31:19Z, approved by "
            "MichAdebayo, Hugging Face revision c6e4cbb7.",
        )
    ]


def test_promotion_and_restore_name_every_version_they_move() -> None:
    script = Path(".github/scripts/promote_model.py").read_text(encoding="utf-8")
    assert script.count('state="production"') >= 2, (
        "promote and restore both name the production version"
    )
    assert 'state="retired"' in script
    assert 'state="rejected"' in script
    assert '"annotate-production"' in script
