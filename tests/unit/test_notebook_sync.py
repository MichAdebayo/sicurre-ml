"""ml/kaggle_training.ipynb is the only training notebook, and the one the workflow ships."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING_NOTEBOOK = ROOT / "ml" / "kaggle_training.ipynb"


def test_the_training_workflow_ships_the_ml_notebook() -> None:
    workflow = (ROOT / ".github" / "workflows" / "train.yml").read_text()
    assert "ml/kaggle_training.ipynb" in workflow
    assert TRAINING_NOTEBOOK.is_file()


def test_no_second_copy_of_the_notebook_sits_at_the_repository_root() -> None:
    """A root copy once drifted from the shipped one; a single source cannot drift."""
    copies = list(ROOT.glob("*.ipynb"))
    assert not copies, "a root notebook copy drifts from ml/kaggle_training.ipynb"
