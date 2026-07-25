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
    workflow = Path(".github/workflows/promote-model.yml").read_text(
        encoding="utf-8"
    )

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
