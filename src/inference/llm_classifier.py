"""Stage 4 — LLM-based classification with tiered fallback.

Tier order: Groq → Cerebras → Perplexity → Gemini
Each tier is tried in order; on failure (rate limit, timeout, error) the next
tier is attempted. Returns None only if all tiers fail.
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass

import httpx


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

    Réponds UNIQUEMENT en JSON avec ce format exact :
    {
      "label": "phishing" | "safe",
      "confidence": <float entre 0.0 et 1.0>,
      "explanation": "<raison courte en français, 1-2 phrases>"
    }
    """
)


def _user_prompt(text: str) -> str:
    return f"Message à analyser :\n\n{text}"


# ---------------------------------------------------------------------------
# Per-provider callers
# ---------------------------------------------------------------------------

def _call_groq(text: str) -> LLMResult | None:
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
        provider="groq",
    )


def _call_cerebras(text: str) -> LLMResult | None:
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
        provider="cerebras",
    )


def _call_perplexity(text: str) -> LLMResult | None:
    api_key = os.getenv("PERPLEXITY_API_KEY")
    model = os.getenv("PERPLEXITY_MODEL", "sonar")
    if not api_key:
        return None
    return _openai_compatible(
        base_url="https://api.perplexity.ai",
        api_key=api_key,
        model=model,
        temperature=0.3,
        text=text,
        provider="perplexity",
    )


def _call_gemini(text: str) -> LLMResult | None:
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    if not api_key:
        return None
    try:
        resp = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": f"{_SYSTEM}\n\n{_user_prompt(text)}"}]}],
                "generationConfig": {"temperature": 0.3},
            },
            timeout=20,
        )
        resp.raise_for_status()
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_response(raw, provider="gemini")
    except Exception as exc:
        print(f"[llm] Gemini error: {exc}")
        return None


def _openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    text: str,
    provider: str,
) -> LLMResult | None:
    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _user_prompt(text)},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=20,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        return _parse_response(raw, provider=provider)
    except Exception as exc:
        print(f"[llm] {provider} error: {exc}")
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
    except Exception as exc:
        print(f"[llm] Parse error from {provider}: {exc} — raw: {raw[:200]}")
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_TIERS = [_call_groq, _call_cerebras, _call_perplexity, _call_gemini]


def classify_llm(text: str) -> LLMResult | None:
    """Try each LLM tier in order; return the first successful result.

    Returns None only if all four providers fail (network outage etc.).
    """
    for tier_fn in _TIERS:
        result = tier_fn(text)
        if result is not None:
            print(
                f"[llm] Response from {result.provider}: "
                f"{result.label} ({result.confidence:.2f})"
            )
            return result
    print("[llm] All tiers failed — LLM stage skipped.")
    return None
