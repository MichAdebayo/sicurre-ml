"""The class weighting must not carry a fixed thumb on the scale.

`inverse_freq` sets w proportional to 1/n, so `n * w` is constant across
classes - the weighting cancels the imbalance exactly, whatever the corpus
looks like. `phishing_boost` is applied to class 0 afterwards, in
SicurreTrainer.compute_loss, and therefore survives that cancellation: phishing
contributes exactly `boost` times the gradient of any other class, no matter
how abundant phishing becomes.

At 2.0 that pull measured 2.00x against legitimate on a corpus where phishing
was 35.3% of the data, and 2.00x again where phishing was 38.1%. Identical,
because the boost does not see the data. It was introduced when phishing was
scarce; by 1 September phishing was the largest class and still carried it.

These tests pin the invariant so the next person changing the boost has to do
it deliberately.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config.training_config import LABEL2ID, TrainingConfig
from src.model.builder import compute_class_weights

PHISHING, SPAM, LEGITIMATE = LABEL2ID["phishing"], LABEL2ID["spam"], LABEL2ID["legitimate"]

#: (phishing, spam, legitimate) - the two real corpora, and a lopsided case.
CORPORA = {
    "18 July (32,591)": (11495, 11576, 9520),
    "1 September (35,151)": (13391, 11909, 9851),
    "phishing-scarce": (2000, 12000, 12000),
}


def _labels(counts: tuple[int, int, int]) -> np.ndarray:
    return np.concatenate([np.full(n, i) for i, n in enumerate(counts)])


@pytest.mark.parametrize("name,counts", CORPORA.items())
def test_inverse_freq_equalises_every_class(name: str, counts: tuple[int, int, int]) -> None:
    """n * w is the same for every class - that is what balancing means."""
    weights = compute_class_weights(_labels(counts), num_labels=3)

    pulls = [n * float(weights[i]) for i, n in enumerate(counts)]
    assert pulls[0] == pytest.approx(pulls[1], rel=1e-6)
    assert pulls[1] == pytest.approx(pulls[2], rel=1e-6)


@pytest.mark.parametrize("name,counts", CORPORA.items())
def test_a_boost_survives_balancing_and_is_blind_to_the_corpus(
    name: str, counts: tuple[int, int, int]
) -> None:
    """A boost of B leaves phishing pulling exactly B times any other class.

    Including the phishing-scarce case: the multiplier is the same there as on
    a corpus where phishing dominates, which is the whole problem.
    """
    weights = compute_class_weights(_labels(counts), num_labels=3).clone()
    boost = 2.0
    weights[PHISHING] *= boost

    phishing_pull = counts[PHISHING] * float(weights[PHISHING])
    legitimate_pull = counts[LEGITIMATE] * float(weights[LEGITIMATE])

    assert phishing_pull / legitimate_pull == pytest.approx(boost, rel=1e-6)


def test_the_shipped_default_leaves_the_classes_balanced() -> None:
    """The default must not silently reintroduce the skew.

    Eight retrains failed the promotion gate the same way - recall up,
    legitimate false positives up harder - which is the signature of a standing
    pull toward phishing.
    """
    assert TrainingConfig().phishing_boost == 1.0

    counts = CORPORA["1 September (35,151)"]
    weights = compute_class_weights(_labels(counts), num_labels=3).clone()
    weights[PHISHING] *= TrainingConfig().phishing_boost

    pulls = [n * float(weights[i]) for i, n in enumerate(counts)]
    assert pulls[PHISHING] == pytest.approx(pulls[LEGITIMATE], rel=1e-6)
    assert pulls[PHISHING] == pytest.approx(pulls[SPAM], rel=1e-6)
