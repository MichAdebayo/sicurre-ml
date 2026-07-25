from __future__ import annotations

import re

_INVALID_UNITY_CATALOG_TAG_CHARACTERS = re.compile(r"[.=><%&?\\]")


def model_version_tag_key(run_tag_key: str) -> str:
    """Return the Unity Catalog-safe equivalent of a dotted MLflow run tag."""
    return _INVALID_UNITY_CATALOG_TAG_CHARACTERS.sub("_", run_tag_key)
