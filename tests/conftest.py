from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Tests never export traces. The serving app configures an OTLP exporter when it
# is imported (alloy:4317 in the Docker stack); a sampled request would try it.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@pytest.fixture(autouse=True)
def reset_service_rate_limiter() -> None:
    from src.serving.rate_limit import service_rate_limiter

    service_rate_limiter.reset()
