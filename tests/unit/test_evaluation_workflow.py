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
    assert "golden-20260719-v1" in script
    assert "bc329213cacddab409a63deb9d663e593351b6e740a45cdada4c201e3beea346" in script
    assert "/internal/ml/candidates" in script
    assert "/internal/ml/evaluations" in script
    assert "workflow_call:" in workflow
    assert "default: production" in workflow
