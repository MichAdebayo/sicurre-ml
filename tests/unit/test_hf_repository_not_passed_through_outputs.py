"""A job output containing a secret arrives empty in the next job.

`HF_USERNAME` is a registered secret, so the string "owner/sicurre-phishing-fr"
contains a secret value. GitHub Actions scrubs job outputs that contain secrets
and passes them downstream as the empty string - not masked, empty.

That silently emptied `hf_repository` on every run. The evaluate job's format
check then failed on "", `set -e` killed the job, and evaluation was skipped.
Automatic candidate evaluation had therefore never succeeded once; the failure
looked like a missing secret and was not.

Every other output in the same block arrived intact, because none of them
contains a secret.

Fix: only the repository name crosses the job boundary. Each consumer
recomposes "owner/name" from the secret it already has.
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


@pytest.mark.parametrize("name,path", WORKFLOWS.items())
def test_no_workflow_passes_owner_slash_name_between_jobs(name: str, path: Path) -> None:
    """`hf_repository` as an input or output is the bug; `hf_repository_name` is the fix."""
    text = path.read_text(encoding="utf-8")

    offenders = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"\bhf_repository\b(?!_name)", line) and not line.strip().startswith("#")
    ]
    assert not offenders, (
        f"{path} still moves the owner-qualified repository between jobs, which "
        f"GitHub empties because it contains HF_USERNAME: {offenders}"
    )


@pytest.mark.parametrize("name", ["evaluate", "promote"])
def test_consumers_recompose_the_repository_from_the_secret(name: str) -> None:
    text = WORKFLOWS[name].read_text(encoding="utf-8")

    assert "HF_REPOSITORY: ${{ secrets.HF_USERNAME }}/${{ inputs.hf_repository_name }}" in text, (
        f"{WORKFLOWS[name]} does not rebuild the full repository from the secret"
    )


def test_the_lineage_step_emits_only_the_name() -> None:
    """Splitting at the producer is what keeps the secret out of the output."""
    text = WORKFLOWS["train"].read_text(encoding="utf-8")

    assert '"hf_repository_name": model["huggingface_repo"].split("/")[-1]' in text


def test_the_format_check_still_validates_a_full_repository() -> None:
    """The recomposed value must still be checked, not trusted.

    An empty HF_USERNAME would produce "/name", which this rejects - so the
    failure stays loud rather than moving somewhere further downstream.
    """
    for name in ("evaluate", "promote"):
        text = WORKFLOWS[name].read_text(encoding="utf-8")
        assert "HF_REPOSITORY" in text
        validates = re.search(r'"\$HF_REPOSITORY"\s*\|', text)
        assert validates, f"{WORKFLOWS[name]} no longer validates the composed repository"
