from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_RETRYABLE = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True, slots=True)
class CallbackResponse:
    id: str
    status: str
    idempotent: bool


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


def post_provenance_callback(
    *,
    base_url: str,
    path: str,
    bearer_token: str,
    payload: dict[str, Any],
    max_attempts: int = 5,
    opener: Callable[..., Any] = urlopen,
    sleeper: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> CallbackResponse:
    """POST a stable callback payload with bounded transient retries."""
    if not path.startswith("/internal/ml/"):
        raise ValueError("Callback path must target the internal ML contract")
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
        },
    )

    for attempt in range(max_attempts):
        try:
            with opener(request, timeout=15) as response:
                result = json.loads(response.read())
                return CallbackResponse(
                    id=str(result["id"]),
                    status=str(result["status"]),
                    idempotent=bool(result["idempotent"]),
                )
        except HTTPError as exc:
            if exc.code not in _RETRYABLE or attempt == max_attempts - 1:
                raise RuntimeError(f"Sicurre callback failed with HTTP {exc.code}") from exc
            retry_after = _retry_after_seconds(exc.headers.get("Retry-After"))
        except (TimeoutError, URLError) as exc:
            if attempt == max_attempts - 1:
                raise RuntimeError("Sicurre callback transport failed") from exc
            retry_after = None
        delay = retry_after if retry_after is not None else (2**attempt + jitter())
        sleeper(delay)

    raise RuntimeError("Sicurre callback exhausted its retry policy")
