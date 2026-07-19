from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.registry import promote_hf_revision


class _FakeApi:
    deleted = False
    created_revision = ""

    def delete_tag(self, **_: str) -> None:
        self.deleted = True

    def create_tag(self, **kwargs: str) -> None:
        self.created_revision = kwargs["revision"]

    def model_info(self, **_: str) -> SimpleNamespace:
        return SimpleNamespace(sha=self.created_revision)


def test_hf_promotion_requires_explicit_approval() -> None:
    with pytest.raises(ValueError, match="Explicit approval"):
        promote_hf_revision("owner/model", "token", "a" * 40, approved=False)


def test_hf_promotion_recreates_and_verifies_tag(monkeypatch) -> None:
    fake_api = _FakeApi()
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: fake_api),
    )

    resolved = promote_hf_revision(
        "owner/model",
        "token",
        "a" * 40,
        approved=True,
    )

    assert fake_api.deleted is True
    assert resolved == "a" * 40
