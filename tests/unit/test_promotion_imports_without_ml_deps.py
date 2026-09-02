"""The promotion path must import without the training stack installed.

`.github/scripts/promote_model.py` runs in a job that installs no ML
dependencies: it moves pointers, checks a Hugging Face revision and posts a
callback. It needs exactly one function from `src.registry.callbacks`, which is
stdlib-only.

Importing that submodule executes `src/registry/__init__.py`, and that file
used to import `src.config.training_config` at module level for two type
annotations. `training_config` imports torch. So a package whose every heavy
import was already deferred inside functions still dragged the entire training
stack into any consumer of any of its modules.

The first promotion ever attempted died on it - `ModuleNotFoundError: No module
named 'torch'` - after the human approval and before any pointer moved. The
ordering meant nothing was left half-changed, but the run was wasted and the
cause was three imports away from the script that failed.

This is exactly the class of defect a test suite misses: every test environment
has torch installed, so nothing here fails when the coupling returns. These
tests assert the shape of the import graph instead of relying on the
environment to be poor enough to notice.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_graph_after(statement: str) -> set[str]:
    """Return sys.modules after running `statement` in a fresh interpreter."""
    code = f"import sys, json\n{statement}\nprint(json.dumps(sorted(sys.modules)))\n"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert result.returncode == 0, (
        f"importing failed, which is the bug this test guards:\n{result.stderr}"
    )
    import json

    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def test_callbacks_import_does_not_pull_in_torch() -> None:
    """The promotion script's only import must stay free of the training stack."""
    modules = _import_graph_after("from src.registry.callbacks import post_provenance_callback")

    assert "torch" not in modules, (
        "src.registry.callbacks now imports torch, so promote_model.py will fail "
        "with ModuleNotFoundError in a job that installs no ML dependencies"
    )
    assert "src.config.training_config" not in modules, (
        "src.registry.__init__ imports training_config at module level again; "
        "keep it under TYPE_CHECKING - the names are annotations only"
    )


def test_registry_package_defers_its_heavy_imports() -> None:
    """Importing the package itself must not cost the training stack either."""
    modules = _import_graph_after("import src.registry")

    for heavy in ("torch", "transformers", "mlflow"):
        assert heavy not in modules, (
            f"src.registry imports {heavy} at module level; move it inside the "
            f"function that needs it, as the rest of this package already does"
        )
