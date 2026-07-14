from __future__ import annotations

import math
import os
import time
from collections import deque
from threading import Lock


def _positive_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


class ServiceRateLimiter:
    """Process-local sliding-window limiter for the single trusted Sicurre caller."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: deque[float] = deque()

    def check(self, now: float | None = None) -> tuple[bool, int]:
        current = time.monotonic() if now is None else now
        sustained_rate = _positive_float("INFERENCE_RATE_LIMIT_RPS", 1.0)
        burst = _positive_int("INFERENCE_RATE_LIMIT_BURST", 5)
        window_seconds = max(1.0, burst / sustained_rate)
        cutoff = current - window_seconds

        with self._lock:
            while self._requests and self._requests[0] <= cutoff:
                self._requests.popleft()
            if len(self._requests) >= burst:
                retry_after = max(1, math.ceil(self._requests[0] + window_seconds - current))
                return False, retry_after
            self._requests.append(current)
        return True, 0

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()


service_rate_limiter = ServiceRateLimiter()
