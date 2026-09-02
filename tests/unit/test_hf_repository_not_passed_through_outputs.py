"""A job output containing a secret arrives empty in the next job.

GitHub Actions scrubs job outputs whose value contains a registered secret and
passes them downstream as the empty string - not masked, empty.

`HF_USERNAME` is a secret, so the string "owner/sicurre-phishing-fr" contains
one. That silently emptied `hf_repository` on every run: the evaluate job's
format check failed on "", `set -e` killed the job, and evaluation was skipped.
Automatic candidate evaluation had therefore never succeeded once, and the
failure looked like a missing secret rather than a scrubbed output.

The first fix passed only the repository *name* across the boundary, on the
theory that the owner was the secret part. That was wrong, and wrong in a way
worth recording: `REPO_NAME` is itself a registered secret, so the name alone
was emptied too and consumers received "***/". It surfaced only on 2 September
2026, on the first candidate in the project's history to pass its gate - nine
earlier candidates failed, and a failing evaluation never reaches the code that
needs the repository.

Splitting a secret does not stop it being a secret. The invariant these tests
pin is therefore stronger than the original one: **nothing about the repository
crosses a job boundary at all**. Each consumer rebuilds "owner/name" from the
two secrets it already has.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOWS = {
    "train": Path(".github/workflows/train.yml"),
    "evaluate": Path(".github/workflows/evaluate-model.yml"),
    "promote": Path(".github/workflows/promote-model.yml"),
}

RECOMPOSED = "HF_REPOSITORY: ${{ secrets.HF_USERNAME }}/${{ secrets.REPO_NAME }}"


@pytest.mark.parametrize("name,path", WORKFLOWS.items())
def test_no_workflow_moves_any_part_of_the_repository_between_jobs(
    name: str, path: Path
) -> None:
    """Neither half may travel as an input or output - both are secrets."""
    text = path.read_text(encoding="utf-8")

    offenders = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"\bhf_repository(_name)?\b", line)
        and not line.strip().startswith("#")
        and "HF_REPOSITORY:" not in line
    ]
    assert not offenders, (
        f"{path} still moves the repository between jobs. Both HF_USERNAME and "
        f"REPO_NAME are registered secrets, so GitHub empties any output "
        f"carrying either: {offenders}"
    )


@pytest.mark.parametrize("name", ["evaluate", "promote"])
def test_consumers_recompose_the_repository_from_both_secrets(name: str) -> None:
    """The value is assembled where it is used, from secrets, never received."""
    text = WORKFLOWS[name].read_text(encoding="utf-8")

    assert RECOMPOSED in text, (
        f"{WORKFLOWS[name]} does not rebuild the repository from both secrets"
    )


@pytest.mark.parametrize("name", ["evaluate", "promote"])
def test_consumers_declare_the_secrets_they_recompose_from(name: str) -> None:
    """A reusable workflow that reads a secret must declare it.

    The callers use `secrets: inherit`, so this works without the declaration -
    which is exactly why it needs a test. A future caller passing secrets
    explicitly would otherwise get an empty owner and a confusing failure far
    from its cause.
    """
    text = WORKFLOWS[name].read_text(encoding="utf-8")

    for secret in ("HF_USERNAME", "REPO_NAME"):
        assert f"{secret}: {{required: true}}" in text, (
            f"{WORKFLOWS[name]} recomposes from {secret} without declaring it"
        )


def test_the_lineage_step_emits_nothing_about_the_repository() -> None:
    """The producer stays silent; there is no safe half to emit."""
    text = WORKFLOWS["train"].read_text(encoding="utf-8")

    assert '"hf_repository_name"' not in text
    assert '"hf_repository"' not in text


def test_the_format_check_still_validates_a_full_repository() -> None:
    """The recomposed value must still be checked, not trusted.

    An empty secret would produce "/name" or "owner/", which this rejects - so
    the failure stays loud rather than moving somewhere further downstream.
    """
    for name in ("evaluate", "promote"):
        text = WORKFLOWS[name].read_text(encoding="utf-8")
        assert "HF_REPOSITORY" in text
        validates = re.search(r'"\$HF_REPOSITORY"\s*\|', text)
        assert validates, f"{WORKFLOWS[name]} no longer validates the composed repository"
