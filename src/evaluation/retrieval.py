from __future__ import annotations

from pathlib import Path

from src.evaluation.golden_set import file_sha256


def download_r2_object(
    *,
    endpoint: str,
    bucket: str,
    object_key: str,
    access_key_id: str,
    secret_access_key: str,
    destination: Path,
    expected_sha256: str,
) -> Path:
    """Download one explicitly named R2 object and verify its immutable checksum."""
    if not object_key.startswith("raw-snapshots/evaluation_sets/"):
        raise ValueError("R2 evaluation object must use the evaluation_sets prefix")
    import boto3
    from botocore.config import Config

    destination.parent.mkdir(parents=True, exist_ok=True)
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="auto",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=Config(signature_version="s3v4"),
    )
    client.download_file(bucket, object_key, str(destination))
    actual = file_sha256(destination)
    if actual != expected_sha256.lower():
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded evaluation checksum mismatch: expected {expected_sha256}, got {actual}"
        )
    return destination
