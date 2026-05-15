from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import load_splits, map_labels, validate_schema


def test_map_labels_converts_string_labels() -> None:
    frame = pd.DataFrame({"text": ["a"], "label": ["phishing"]})
    mapped = map_labels(frame, {"phishing": 0, "spam": 1, "legitimate": 2})
    assert mapped.loc[0, "label"] == 0


def test_validate_schema_rejects_missing_text() -> None:
    frame = pd.DataFrame({"label": [0]})
    with pytest.raises(ValueError):
        validate_schema(frame, "train")


def test_load_splits_reads_expected_csvs(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        pd.DataFrame({"text": ["email"], "label": ["spam"]}).to_csv(
            tmp_path / f"sicurre_{split}.csv",
            index=False,
        )
    splits = load_splits(tmp_path, {"phishing": 0, "spam": 1, "legitimate": 2})
    assert len(splits.train) == 1
    assert len(splits.val) == 1
    assert len(splits.test) == 1
