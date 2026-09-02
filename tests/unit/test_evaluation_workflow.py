from pathlib import Path


def test_evaluation_workflow_uses_exact_object_and_secret_contract() -> None:
    workflow = Path(".github/workflows/evaluate-model.yml").read_text()
    script = Path(".github/scripts/evaluate_candidate.py").read_text()

    for name in (
        "R2_EVALUATION_ACCESS_KEY_ID",
        "R2_EVALUATION_SECRET_ACCESS_KEY",
        "R2_EVALUATION_ENDPOINT",
        "R2_EVALUATION_BUCKET_NAME",
        "SICURRE_CALLBACK_BASE_URL",
        "SICURRE_INTERNAL_API_KEY",
    ):
        assert name in workflow
    # The evaluation set is still pinned by version and checksum - the pin just
    # moved from literals here to GOLDEN_SET_RELEASES, so that publishing a new
    # set updates the gate instead of leaving it on an old one. v3 sat in R2
    # unused from 16 August while the gate scored against a July set.
    from src.evaluation.golden_set import GOLDEN_SET_RELEASES, latest_golden_set

    assert "latest_golden_set()" in script, (
        "the evaluation script no longer resolves the newest registered golden set"
    )
    selected = latest_golden_set()
    assert selected.version and len(selected.sha256) == 64, (
        "the selected golden set is not fully pinned by version and checksum"
    )
    # Every historical set stays registered, so an old evaluation run's recorded
    # version still resolves to the bytes it was scored against.
    registered = {release.version for release in GOLDEN_SET_RELEASES}
    assert "golden-20260719-v1" in registered
    assert any(
        r.sha256 == "bc329213cacddab409a63deb9d663e593351b6e740a45cdada4c201e3beea346"
        for r in GOLDEN_SET_RELEASES
    )
    assert "/internal/ml/candidates" in script
    assert "/internal/ml/evaluations" in script
    assert "workflow_call:" in workflow
    assert "default: production" in workflow
    assert "evaluation_run_id:" in workflow
    assert "outcome:" in workflow
    assert "GITHUB_OUTPUT" in script
    assert "sicurre.candidate.mlflow_model_version" in script
    assert '"failed": "rejected"' in script
