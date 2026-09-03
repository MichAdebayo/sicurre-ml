"""The recall margin is derived from the golden set, so the two must agree.

`PromotionThresholds.recall_regression_tolerance` is 0.099 because that is the
Wilson 95% half-width of a proportion estimated on **42 phishing samples**. It is
not a policy preference; it is a property of the measuring instrument.

That makes it fragile in a way nothing else catches. Publishing a new evaluation
set with the same total but a different class balance changes the correct margin
while leaving the constant untouched, and every test still passes. The gate would
then admit or reject candidates against a number derived from a set that no
longer exists.

These tests recompute the margin from the registered class counts and assert the
constant still matches, so a new set fails loudly instead of silently shifting
the bar.
"""

from __future__ import annotations

from math import sqrt

from src.evaluation.golden_set import GOLDEN_SET_RELEASES, latest_golden_set
from src.evaluation.promotion import PromotionThresholds

#: The incumbent's phishing recall at the time the margin was derived, 37/42.
#: The half-width depends on the proportion as well as the sample size, so the
#: derivation needs both.
_INCUMBENT_PHISHING_RECALL = 37 / 42


def _wilson_half_width(proportion: float, n: int, z: float = 1.96) -> float:
    """Half-width of the Wilson score interval — the derivation, not a lookup."""
    denominator = 1 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    spread = z * sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n))
    half = spread / denominator
    # centre is unused for the half-width itself but is what the interval is
    # centred on; referencing it keeps the formula readable as a whole.
    assert 0 <= centre <= 1
    return half


def test_class_counts_sum_to_the_recorded_total() -> None:
    """A registered set that does not add up is a transcription error."""
    for release in GOLDEN_SET_RELEASES:
        assert sum(release.class_counts.values()) == release.sample_count, (
            f"{release.version}: class counts {dict(release.class_counts)} do not "
            f"sum to sample_count {release.sample_count}"
        )


def test_every_release_records_all_three_classes() -> None:
    """The gate scores three classes; a set missing one cannot support it."""
    for release in GOLDEN_SET_RELEASES:
        assert set(release.class_counts) == {"phishing", "legitimate", "spam"}, (
            f"{release.version} records {sorted(release.class_counts)}"
        )


def test_recall_margin_matches_the_current_golden_set() -> None:
    """0.099 must remain the half-width for the live set's phishing count.

    If this fails after publishing a new set, the margin is stale rather than
    the test being wrong: re-derive it from the new phishing count and update
    both `PromotionThresholds` and the mirrored constant in the companion
    repository, which re-derives the gate as a cross-check.
    """
    phishing = latest_golden_set().class_counts["phishing"]
    expected = _wilson_half_width(_INCUMBENT_PHISHING_RECALL, phishing)
    configured = PromotionThresholds().recall_regression_tolerance

    assert abs(configured - expected) < 0.0005, (
        f"the margin is {configured} but {phishing} phishing samples give a "
        f"Wilson half-width of {expected:.4f}. The golden set changed without "
        f"the margin being re-derived, so the gate is measuring against a set "
        f"that no longer exists."
    )


def test_a_larger_set_would_tighten_the_margin() -> None:
    """The margin must shrink as the set grows, or it is not derived from it.

    This is the property that makes the margin defensible rather than arbitrary:
    it is a statement about measurement precision, so more samples must make it
    stricter without anyone deciding to.
    """
    current = latest_golden_set().class_counts["phishing"]
    wider = _wilson_half_width(_INCUMBENT_PHISHING_RECALL, current)
    tighter = _wilson_half_width(_INCUMBENT_PHISHING_RECALL, current * 4)

    assert tighter < wider, "quadrupling the sample count must narrow the interval"
