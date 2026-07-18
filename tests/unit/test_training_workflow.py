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
