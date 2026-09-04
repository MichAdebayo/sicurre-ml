"""The manifest must show which provider models the process will call.

A decommissioned Groq model went unnoticed while every classification silently
degraded to ONNX alone, because the only way to check whether an env change had
reached the running container was shell access to the host. The manifest is the
endpoint that answers "what is this process actually configured to do", and it
did not answer that for the LLM chain.

Model names are not secrets. API keys are, so the manifest reports only whether
one is present.
"""

from __future__ import annotations

import json

import pytest

from src.serving.identity import deployment_manifest

_KEYS = ("GROQ_API_KEY", "MISTRAL_API_KEY", "CEREBRAS_API_KEY")
_MODELS = ("GROQ_MODEL", "MISTRAL_MODEL", "CEREBRAS_MODEL")


@pytest.fixture
def clean_env(monkeypatch):
    for name in _KEYS + _MODELS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_the_configured_groq_model_is_visible(clean_env) -> None:
    clean_env.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    assert deployment_manifest()["llm"]["groq_model"] == "openai/gpt-oss-20b"


def test_an_unset_model_reads_as_default_not_as_blank(clean_env) -> None:
    """Blank would look like a configured empty value; it is not the same thing."""
    assert deployment_manifest()["llm"]["groq_model"] == "(default)"


def test_only_configured_providers_are_listed(clean_env) -> None:
    clean_env.setenv("GROQ_API_KEY", "secret-value")
    llm = deployment_manifest()["llm"]
    assert llm["providers_configured"] == ["groq"]


def test_api_keys_never_appear_in_the_manifest(clean_env) -> None:
    """The manifest is served to authenticated callers; a key must not leak."""
    for name in _KEYS:
        clean_env.setenv(name, f"secret-{name}")
    clean_env.setenv("GROQ_MODEL", "openai/gpt-oss-20b")

    rendered = json.dumps(deployment_manifest())
    for name in _KEYS:
        assert f"secret-{name}" not in rendered, f"{name} leaked into the manifest"
    assert sorted(deployment_manifest()["llm"]["providers_configured"]) == [
        "cerebras",
        "groq",
        "mistral",
    ]


def test_the_manifest_still_carries_model_identity(clean_env) -> None:
    """Adding the llm block must not disturb what the CD validator checks."""
    manifest = deployment_manifest()
    assert manifest["schema_version"] == 1
    for section in ("service", "model", "dataset", "deployment", "llm"):
        assert section in manifest
