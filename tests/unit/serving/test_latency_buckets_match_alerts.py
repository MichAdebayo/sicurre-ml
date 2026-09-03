"""A latency alert must ask a question the histogram can answer.

`histogram_quantile` on a classic histogram returns the highest finite boundary
for any quantile landing in the overflow bucket. So a threshold above the top
boundary is unreachable: the query cannot produce a value that exceeds it, and
the alert can never fire however slow the service becomes.

That was the state until 3 September 2026. Buckets ended at 5000 ms and the
deployed rule asked whether p95 exceeded 8000 ms. The rule looked healthy on
every dashboard because it was unfalsifiable, not because latency was good.

These tests bind the two together so the pair cannot drift apart again. They are
deliberately about the relationship rather than the values: change the objective,
change both, and they still pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.serving.telemetry import _PROMETHEUS_BUCKETS_MS

ALERTS = Path("deploy/grafana/alerts/sicurre-ml-alerts.json")


def _latency_alert_thresholds() -> dict[str, float]:
    """Thresholds of every alert whose title names a latency percentile."""
    raw = ALERTS.read_text(encoding="utf-8")
    alerts = json.loads(raw)
    blob = json.dumps(alerts)

    found: dict[str, float] = {}
    for match in re.finditer(
        r'"title":\s*"(?P<title>[^"]*inference p95 above (?P<value>[0-9.]+)s)"', blob
    ):
        found[match.group("title")] = float(match.group("value")) * 1000
    return found


def test_every_latency_alert_is_within_the_histogram_range() -> None:
    """No threshold may sit above the highest finite boundary."""
    ceiling = max(_PROMETHEUS_BUCKETS_MS)
    thresholds = _latency_alert_thresholds()

    assert thresholds, "no latency alert found — the pattern or the titles changed"

    for title, threshold_ms in thresholds.items():
        assert threshold_ms <= ceiling, (
            f"{title!r} asks for {threshold_ms:.0f} ms but the histogram tops out "
            f"at {ceiling:.0f} ms, so the quantile can never reach it and the "
            f"alert cannot fire"
        )


def test_alert_thresholds_land_on_a_bucket_boundary() -> None:
    """A threshold between boundaries is reported at the boundary below it.

    Interpolation inside a bucket is an estimate; a threshold placed on a real
    boundary means the alert fires on a value the histogram actually records.
    """
    thresholds = _latency_alert_thresholds()

    for title, threshold_ms in thresholds.items():
        assert threshold_ms in _PROMETHEUS_BUCKETS_MS, (
            f"{title!r} uses {threshold_ms:.0f} ms, which is not a bucket "
            f"boundary; boundaries are {sorted(_PROMETHEUS_BUCKETS_MS)}"
        )


def test_buckets_resolve_the_band_the_objective_cares_about() -> None:
    """At least three boundaries between 500 ms and the ceiling.

    A single boundary across the objective band makes "comfortably inside" and
    "about to breach" the same observation. The previous set had exactly one
    between 1000 and 5000 ms.
    """
    ceiling = max(_PROMETHEUS_BUCKETS_MS)
    in_band = [b for b in _PROMETHEUS_BUCKETS_MS if 500 < b <= ceiling]

    assert len(in_band) >= 3, (
        f"only {len(in_band)} boundaries above 500 ms: {in_band}. The objective "
        f"band needs resolution, not a single wide bucket."
    )


def test_the_ceiling_is_reachable_by_the_configured_timeouts() -> None:
    """The top boundary must exceed what the chain can actually produce.

    The LLM chain is bounded at 1.5 s and the caller aborts at 2.5 s, so a
    request cannot legitimately exceed roughly 2.5 s. A ceiling below that would
    make the overflow bucket the normal home for slow requests; far above it
    wastes resolution on durations that cannot occur.
    """
    ceiling = max(_PROMETHEUS_BUCKETS_MS)

    assert 2000 <= ceiling <= 5000, (
        f"ceiling {ceiling:.0f} ms does not bracket the configured timeout "
        f"budget; re-cut the buckets when the budget changes"
    )
