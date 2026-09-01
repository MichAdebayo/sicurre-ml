"""Training must be reproducible before any promotion comparison means anything.

Two runs of this pipeline on the same dataset, the same four epochs and the same
168 recorded parameters produced phishing recall of 0.68 and 0.92, and legitimate
false-positive rates of 0.28 and 0.56. The promotion gate permits no regression
on either, so it was comparing draws rather than models.
"""

import os
import random

import numpy as np
import torch

from src.config.training_config import enable_deterministic_training


def test_reports_every_knob_it_pinned() -> None:
    """The returned mapping is logged to MLflow, so the guarantee is auditable."""
    info = enable_deterministic_training(42)

    assert info["seed"] == 42
    assert info["cudnn_deterministic"] is True
    assert info["cudnn_benchmark"] is False
    assert info["deterministic_algorithms"] is True
    assert info["cublas_workspace_config"] == ":4096:8"


def test_cudnn_stops_benchmarking_algorithms() -> None:
    """cuDNN otherwise keeps whichever algorithm was fastest on that machine."""
    enable_deterministic_training(42)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_cublas_workspace_is_configured() -> None:
    """Deterministic CUBLAS reductions require this before the CUDA context."""
    enable_deterministic_training(42)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_all_three_generators_are_reseeded_together() -> None:
    """Seeding torch alone leaves numpy and random free to diverge."""
    enable_deterministic_training(42)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    enable_deterministic_training(42)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert first == second


def test_a_different_seed_produces_different_draws() -> None:
    """Guards against a helper that pins everything to a constant regardless."""
    enable_deterministic_training(42)
    with_42 = float(torch.rand(1))

    enable_deterministic_training(7)
    with_7 = float(torch.rand(1))

    assert with_42 != with_7
