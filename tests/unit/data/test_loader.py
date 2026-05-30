from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import load_splits, map_labels, summarize_split, validate_schema


def test_map_labels_converts_string_labels() -> None:
    frame = pd.DataFrame({"text": ["a"], "label": ["phishing"]})
    mapped = map_labels(frame, {"phishing": 0, "spam": 1, "legitimate": 2})
    assert mapped.loc[0, "label"] == 0


def test_validate_schema_rejects_missing_text() -> None:
    frame = pd.DataFrame({"label": [0]})
    with pytest.raises(ValueError):
        validate_schema(frame, "train")


def test_validate_schema_rejects_null_text() -> None:
    frame = pd.DataFrame({"text": [None], "label": [0]})
    with pytest.raises(ValueError, match="null text"):
        validate_schema(frame, "train")


def test_validate_schema_rejects_invalid_label() -> None:
    frame = pd.DataFrame({"text": ["hello"], "label": [99]})
    with pytest.raises(ValueError, match="unexpected label"):
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


def test_summarize_split_returns_correct_structure() -> None:
    frame = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 0]})
    result = summarize_split(frame, {0: "phishing", 1: "spam", 2: "legitimate"})
    assert result["rows"] == 3
    assert result["distribution"]["phishing"] == 2
    assert result["distribution"]["spam"] == 1

