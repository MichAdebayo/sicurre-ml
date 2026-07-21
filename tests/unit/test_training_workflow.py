import json
from pathlib import Path


def _workflow() -> str:
    return Path(".github/workflows/train.yml").read_text(encoding="utf-8")


def test_training_dataset_does_not_cross_job_output_boundary() -> None:
    workflow = _workflow()
    validation_job = workflow.split("  validate-dispatch:", maxsplit=1)[1].split(
        "  sync-and-retrain:", maxsplit=1
    )[0]

    assert "outputs:" not in validation_job
    assert "GITHUB_OUTPUT" not in validation_job
    assert "needs.validate-dispatch.outputs.training_dataset" not in workflow
    assert "KAGGLE_TRAINING_DATASET: ${{ inputs.training_dataset }}" in workflow


def test_real_training_requires_main_and_revalidates_dataset() -> None:
    workflow = _workflow()
    training_job = workflow.split("  sync-and-retrain:", maxsplit=1)[1]
    guard = training_job.split("      - name: Checkout", maxsplit=1)[0]

    assert 'if [ "$GITHUB_REF_NAME" != "main" ]' in guard
    assert 'if [ -z "$KAGGLE_TRAINING_DATASET" ]' in guard
    assert "^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$" in guard
    assert '"${KAGGLE_TRAINING_DATASET}"' in training_job


def test_validate_only_never_starts_training_job() -> None:
    workflow = _workflow()
    training_job = workflow.split("  sync-and-retrain:", maxsplit=1)[1]

    assert "inputs.validate_only != true" in training_job


def test_training_checks_canonical_kaggle_splits_before_kernel_push() -> None:
    workflow = _workflow()
    training_job = workflow.split("  sync-and-retrain:", maxsplit=1)[1]
    preflight_position = training_job.index("Validate Kaggle training dataset files")
    kernel_push_position = training_job.index("kaggle kernels push")

    assert preflight_position < kernel_push_position
    assert 'kaggle datasets files "$KAGGLE_TRAINING_DATASET" -v' in training_job
    assert "for expected_file in train.csv val.csv test.csv" in training_job


def test_dispatch_requires_immutable_dataset_lineage() -> None:
    workflow = _workflow()

    for field in (
        "dataset_id:",
        "dataset_version:",
        "dataset_sha256:",
    ):
        assert field in workflow
    assert "dataset_sha256 must be exactly 64 hexadecimal characters" in workflow
    assert "MODEL_SEMANTIC_VERSION: 0.0.0-candidate.${{ github.run_id }}" in workflow


def test_notebook_publishes_candidate_without_automatic_production_move() -> None:
    notebook = json.loads(
        Path("ml/kaggle_training.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "stage_candidate" in source
    assert "publish_candidate_to_hub" in source
    assert "new_training_manifest" in source
    assert "promote_if_threshold" not in source
    assert "Promoted to @production" not in source


def test_successful_training_automatically_calls_golden_evaluation() -> None:
    workflow = _workflow()

    assert "Download and validate candidate lineage manifest" in workflow
    assert "training-manifest.json" in workflow
    assert "evaluate-candidate:" in workflow
    assert "uses: ./.github/workflows/evaluate-model.yml" in workflow
    assert "needs.sync-and-retrain.result == 'success'" in workflow
    assert "incumbent_hf_revision: production" in workflow
