from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation.retrieval import download_r2_object


class _FakeS3:
    def download_file(self, bucket: str, key: str, destination: str) -> None:
        assert bucket == "sicurre-raw"
        assert key.endswith("golden.jsonl")
        Path(destination).write_bytes(b"approved")


def test_downloads_exact_object_and_verifies_checksum(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda *args, **kwargs: _FakeS3()),
    )
    monkeypatch.setitem(
        sys.modules,
        "botocore.config",
        SimpleNamespace(Config=lambda **kwargs: kwargs),
    )
    destination = tmp_path / "golden.jsonl"

    result = download_r2_object(
        endpoint="https://account.r2.cloudflarestorage.com",
        bucket="sicurre-raw",
        object_key="golden.jsonl",
        access_key_id="key",
        secret_access_key="secret",
        destination=destination,
        expected_sha256=hashlib.sha256(b"approved").hexdigest(),
    )

    assert result.read_bytes() == b"approved"


def test_checksum_failure_removes_download(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda *a, **k: _FakeS3()))
    monkeypatch.setitem(
        sys.modules,
        "botocore.config",
        SimpleNamespace(Config=lambda **kwargs: kwargs),
    )
    destination = tmp_path / "golden.jsonl"

    with pytest.raises(ValueError, match="checksum mismatch"):
        download_r2_object(
            endpoint="https://account.r2.cloudflarestorage.com",
            bucket="sicurre-raw",
            object_key="golden.jsonl",
            access_key_id="key",
            secret_access_key="secret",
            destination=destination,
            expected_sha256="0" * 64,
        )

    assert not destination.exists()
