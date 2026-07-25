from pathlib import Path


def test_local_and_kaggle_training_notebooks_are_byte_identical() -> None:
    assert Path("sicurre-camembertv2-finetuning.ipynb").read_bytes() == Path(
        "ml/kaggle_training.ipynb"
    ).read_bytes()
