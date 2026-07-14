"""Stage 4 — LLM-based classification with tiered fallback.

Tier order: Groq → Cerebras → Gemini
Each tier is tried in order; on failure (rate limit, timeout, error) the next
tier is attempted. Returns None only if all tiers fail.
"""

from __future__ import annotations

import json
import os
import random
import textwrap
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Mapping

import httpx

_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
_circuit_lock = Lock()
_circuit_failures: dict[str, int] = {}
_circuit_opened_at: dict[str, float] = {}


@dataclass
class LLMResult:
    label: str          # "phishing" or "safe"
    confidence: float   # 0.0–1.0
    explanation: str
    provider: str       # which tier responded


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = textwrap.dedent(
    """\
    Tu es un expert en cybersécurité spécialisé dans la détection de phishing
    en français. Analyse le message fourni et détermine s'il s'agit d'une
    tentative de phishing (hameçonnage) ou d'un message légitime.

    Tu dois impérativement évaluer:
    - l'adresse expéditeur (signes de spoofing, domaine suspect, typosquatting),
    - l'objet (urgence anormale, menaces, demande de paiement, réinitialisation),
    - le corps du message (liens, pression psychologique, incohérences).

    Si une vérification web externe n'est pas disponible, indique explicitement
    que la réputation externe n'a pas pu être vérifiée dans l'explication.

    Réponds UNIQUEMENT en JSON avec ce format exact :
    {
      "label": "phishing" | "safe",
      "confidence": <float entre 0.0 et 1.0>,
      "explanation": "<raison courte en français, 1-2 phrases>"
    }
    """
)


def _user_prompt(text: str, sender: str | None = None, subject: str | None = None) -> str:
    sender_value = sender.strip() if sender else "(non fourni)"
    subject_value = subject.strip() if subject else "(non fourni)"
    return (
        "Email à analyser :\n"
        f"Expéditeur: {sender_value}\n"
        f"Objet: {subject_value}\n"
        f"Corps:\n{text}"
    )


# ---------------------------------------------------------------------------
# Per-provider callers
# ---------------------------------------------------------------------------

def _call_groq(
    text: str,
    sender: str | None = None,
    subject: str | None = None,
) -> LLMResult | None:
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    temperature = float(os.getenv("GROQ_MODEL_TEMPERATURE", "0.3"))
    if not api_key:
        return None
    return _openai_compatible(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        model=model,
        temperature=temperature,
        text=text,
        sender=sender,
        subject=subject,
        provider="groq",
    )


def _call_cerebras(
    text: str,
    sender: str | None = None,
    subject: str | None = None,
) -> LLMResult | None:
    api_key = os.getenv("CEREBRAS_API_KEY")
    model = os.getenv("CEREBRAS_MODEL", "llama3.1-8b")
    temperature = float(os.getenv("CEREBRAS_MODEL_TEMPERATURE", "0.3"))
    if not api_key:
        return None
    return _openai_compatible(
        base_url="https://api.cerebras.ai/v1",
        api_key=api_key,
        model=model,
        temperature=temperature,
        text=text,
        sender=sender,
        subject=subject,
        provider="cerebras",
    )


def _call_gemini(
    text: str,
    sender: str | None = None,
    subject: str | None = None,
) -> LLMResult | None:
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    if not api_key:
        return None
    try:
        resp = _resilient_post(
            provider="gemini",
            url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": api_key},
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": (
                                    f"{_SYSTEM}\n\n"
                                    f"{_user_prompt(text, sender=sender, subject=subject)}"
                                )
                            }
                        ]
                    }
                ],
                "generationConfig": {"temperature": 0.3},
            },
        )
        resp.raise_for_status()
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_response(raw, provider="gemini")
    except Exception:
        _emit_provider_event("gemini", "request_failed")
        return None


def _openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    text: str,
    sender: str | None = None,
    subject: str | None = None,
    provider: str,
) -> LLMResult | None:
    try:
        resp = _resilient_post(
            provider=provider,
            url=f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {
                        "role": "user",
                        "content": _user_prompt(text, sender=sender, subject=subject),
                    },
                ],
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        return _parse_response(raw, provider=provider)
    except Exception:
        _emit_provider_event(provider, "request_failed")
        return None


def _parse_response(raw: str, provider: str) -> LLMResult | None:
    try:
        # strip markdown code fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
            clean = clean.rstrip("`").strip()
        data = json.loads(clean)
        label = str(data.get("label", "safe")).lower()
        if label not in ("phishing", "safe"):
            label = "safe"
        confidence = float(data.get("confidence", 0.5))
        explanation = str(data.get("explanation", ""))
        return LLMResult(
            label=label,
            confidence=max(0.0, min(1.0, confidence)),
            explanation=explanation,
            provider=provider,
        )
    except Exception:
        _emit_provider_event(provider, "invalid_response")
        return None


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.05, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _emit_provider_event(provider: str, category: str) -> None:
    print(
        json.dumps(
            {"event": "llm_provider", "provider": provider, "category": category},
            sort_keys=True,
        ),
        flush=True,
    )


def _circuit_allows(provider: str, now: float) -> bool:
    cooldown = _env_float("LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS", 60.0)
    with _circuit_lock:
        opened_at = _circuit_opened_at.get(provider)
        if opened_at is None:
            return True
        if now - opened_at >= cooldown:
            _circuit_opened_at.pop(provider, None)
            _circuit_failures[provider] = 0
            return True
        return False


def _record_provider_result(provider: str, *, success: bool, now: float) -> None:
    threshold = _env_int("LLM_CIRCUIT_BREAKER_FAILURES", 3)
    with _circuit_lock:
        if success:
            _circuit_failures[provider] = 0
            _circuit_opened_at.pop(provider, None)
            return
        failures = _circuit_failures.get(provider, 0) + 1
        _circuit_failures[provider] = failures
        if failures >= threshold:
            _circuit_opened_at[provider] = now


def _resilient_post(
    provider: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
    json: Any = None,
) -> httpx.Response:
    now = time.monotonic()
    if not _circuit_allows(provider, now):
        _emit_provider_event(provider, "circuit_open")
        raise RuntimeError("LLM provider circuit is open")

    attempts = _env_int("LLM_MAX_ATTEMPTS", 2)
    connect_timeout = _env_float("LLM_CONNECT_TIMEOUT_SECONDS", 3.0)
    response_timeout = _env_float("LLM_RESPONSE_TIMEOUT_SECONDS", 12.0)
    backoff = _env_float("LLM_RETRY_BACKOFF_SECONDS", 0.25)
    timeout = httpx.Timeout(response_timeout, connect=connect_timeout)

    for attempt in range(attempts):
        try:
            response = httpx.post(
                url,
                headers=headers,
                params=params,
                json=json,
                timeout=timeout,
            )
            if response.status_code not in _RETRYABLE_STATUS_CODES:
                _record_provider_result(provider, success=True, now=time.monotonic())
                return response
            retryable = True
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout):
            retryable = True

        if attempt + 1 < attempts and retryable:
            delay = backoff * (2**attempt) + random.uniform(0.0, backoff)
            time.sleep(delay)

    _record_provider_result(provider, success=False, now=time.monotonic())
    raise RuntimeError("LLM provider exhausted bounded retries")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_TIERS = [_call_groq, _call_cerebras, _call_gemini]


def classify_llm(
    text: str,
    *,
    sender: str | None = None,
    subject: str | None = None,
) -> LLMResult | None:
    """Try each LLM tier in order; return the first successful result.

    Returns None only if all four providers fail (network outage etc.).
    """
    for tier_fn in _TIERS:
        result = tier_fn(text, sender=sender, subject=subject)
        if result is not None:
            print(
                f"[llm] Response from {result.provider}: "
                f"{result.label} ({result.confidence:.2f})"
            )
            return result
    print("[llm] All tiers failed — LLM stage skipped.")
    return None
