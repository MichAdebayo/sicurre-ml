"""The candidate download must survive a transient network fault.

`kaggle kernels output` pulls roughly 890MB back from Kaggle *after* the GPU
training has already finished. It ran exactly once. On 1 September the kernel
reported COMPLETE with no failure message and the workflow still failed on

    Connection broken: IncompleteRead(230600704 bytes read,
                                      659244026 more expected)

throwing away a finished model, skipping evaluation and promotion entirely.
The candidate had to be recovered from Kaggle by hand.

The asymmetry is what makes this worth a test: everything expensive has already
succeeded by the time this step runs, so a single dropped connection is the
most costly possible place to have no retry.
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(".github/workflows/train.yml")


def _download_step() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    step = text.split("Download and validate candidate lineage manifest", maxsplit=1)[1]
    return step.split("- name:", maxsplit=1)[0]


def test_the_kernel_output_download_is_retried() -> None:
    step = _download_step()

    assert "kaggle kernels output" in step, "download step no longer present"

    match = re.search(r"for attempt in ([\d ]+);", step)
    assert match, (
        "the kernel output download is not wrapped in a retry loop; a dropped "
        "connection here discards a model that has already finished training"
    )
    # `for attempt in 1` is a loop that retries nothing, and reads as a retry.
    attempts = match.group(1).split()
    assert len(attempts) >= 3, (
        f"retry loop makes only {len(attempts)} attempt(s); a single transient "
        f"IncompleteRead should not cost a completed training run"
    )


def test_a_failed_download_does_not_pass_silently() -> None:
    """Retrying and then continuing on failure would be worse than not retrying."""
    step = _download_step()

    assert "exit 1" in step, "an exhausted download must fail the job"


def test_each_attempt_starts_from_a_clean_directory() -> None:
    """A partial download left in place can satisfy the manifest lookup.

    `find -name training-manifest.json` would then succeed against the debris of
    a truncated attempt, and the job would proceed with an incomplete model
    directory.
    """
    step = _download_step()

    assert "rm -rf" in step, "retry does not clear the partial download"


def test_the_operator_is_told_the_model_still_exists() -> None:
    """It is recoverable, and the error message is where that gets noticed."""
    step = _download_step()

    assert "may still exist on Kaggle" in step
