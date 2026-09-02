"""The gate must score against the newest published evaluation set.

`golden-20260816-v3` was published to R2 on 16 August and sat unused. The gate
went on scoring candidates against a 60-sample set from 19 July, because the
version, key and checksum were pinned as constants in the evaluation script.

That matters beyond tidiness. On 60 samples one legitimate email flipping moves
the false-positive rate by 0.04, and one email was the entire margin between the
incumbent and candidate 1.0.27. A gate that cannot grow gets weaker relative to
the decision it is being asked to make.

The registry stays explicit rather than discovered from storage: the gate decides
what reaches production, so listing a bucket and trusting whatever is newest
would let anyone with write access to that bucket move the bar.
"""

from __future__ import annotations

import pytest

from src.evaluation.golden_set import (
    GOLDEN_SET_RELEASES,
    latest_golden_set,
)


def test_the_latest_release_is_the_one_selected() -> None:
    assert latest_golden_set() is GOLDEN_SET_RELEASES[-1]


def test_the_registry_is_ordered_oldest_to_newest() -> None:
    """`latest` takes the last entry, so ordering is load-bearing, not cosmetic.

    Version strings carry the publication date, so lexical order is chronological
    order for this naming scheme.
    """
    versions = [release.version for release in GOLDEN_SET_RELEASES]
    assert versions == sorted(versions), (
        f"GOLDEN_SET_RELEASES must be ordered oldest first; got {versions}"
    )


def test_the_selected_set_is_the_larger_august_set_not_the_july_one() -> None:
    """The regression this exists to prevent, stated concretely."""
    latest = latest_golden_set()

    assert latest.version == "golden-20260816-v3"
    assert latest.sample_count == 95
    assert latest.object_key == "evaluation_sets/golden-20260816-v3/golden.jsonl"


def test_every_release_is_fully_pinned() -> None:
    """Each entry names an immutable object and its checksum.

    Without the checksum the gate would accept whatever currently sits at that
    key, which is the property the retrieval allowlist exists to protect.
    """
    for release in GOLDEN_SET_RELEASES:
        assert release.version
        assert release.object_key
        assert len(release.sha256) == 64, f"{release.version}: sha256 is not a full digest"
        assert release.sha256 == release.sha256.lower()
        assert release.schema_version
        assert release.sample_count > 0


def test_versions_are_unique() -> None:
    versions = [release.version for release in GOLDEN_SET_RELEASES]
    assert len(versions) == len(set(versions))


def test_retrieval_accepts_every_registered_key_and_nothing_else() -> None:
    """The allowlist and the registry must not drift apart.

    A set registered but not retrievable fails at promotion time, on the run
    that needed it.
    """
    from src.evaluation.retrieval import download_r2_object

    with pytest.raises(ValueError, match="restricted to registered golden sets"):
        download_r2_object(
            endpoint="https://example.invalid",
            bucket="b",
            object_key="some/other/object.jsonl",
            access_key_id="k",
            secret_access_key="s",
            destination=__import__("pathlib").Path("/tmp/never-written.jsonl"),
            expected_sha256="0" * 64,
        )


def test_an_empty_registry_fails_loudly() -> None:
    import src.evaluation.golden_set as module

    original = module.GOLDEN_SET_RELEASES
    try:
        module.GOLDEN_SET_RELEASES = ()
        with pytest.raises(ValueError, match="No golden set releases"):
            module.latest_golden_set()
    finally:
        module.GOLDEN_SET_RELEASES = original


def test_a_release_is_immutable() -> None:
    """Frozen so nothing can rewrite the bar at run time."""
    release = latest_golden_set()
    with pytest.raises((AttributeError, TypeError)):
        release.sha256 = "0" * 64  # type: ignore[misc]
