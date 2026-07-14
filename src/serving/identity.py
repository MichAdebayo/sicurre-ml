from __future__ import annotations

import os
from typing import Any

from src.inference.onnx_classifier import get_model_version


def deployment_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "service": {
            "name": "sicurre-ml",
            "version": os.getenv("SERVICE_VERSION", "0.1.0"),
            "api_contract": "v1",
        },
        "model": {
            "version": os.getenv("MODEL_VERSION", "phishing-detector-1.0.0"),
            "revision": get_model_version(),
        },
        "dataset": {
            "version": os.getenv("DATASET_VERSION", "unknown"),
        },
        "deployment": {
            "revision": os.getenv("DEPLOYMENT_REVISION", os.getenv("IMAGE_TAG", "unknown")),
            "container_image_digest": os.getenv("CONTAINER_IMAGE_DIGEST", "unknown"),
        },
    }


def response_identity_headers() -> dict[str, str]:
    manifest = deployment_manifest()
    return {
        "X-Sicurre-Service-Version": str(manifest["service"]["version"]),
        "X-Sicurre-Model-Version": str(manifest["model"]["version"]),
        "X-Sicurre-Model-Revision": str(manifest["model"]["revision"]),
        "X-Sicurre-Deployment-Revision": str(manifest["deployment"]["revision"]),
    }
