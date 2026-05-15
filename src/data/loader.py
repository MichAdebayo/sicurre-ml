from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

REQUIRED_COLUMNS = {"text", "label"}


@dataclass(slots=True)
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def map_labels(frame: pd.DataFrame, label2id: dict[str, int]) -> pd.DataFrame:
    mapped = frame.copy()
    if not is_numeric_dtype(mapped["label"]):
        mapped["label"] = mapped["label"].map(label2id)
    return mapped


def validate_schema(
    frame: pd.DataFrame,
    split_name: str,
    valid_labels: tuple[int, ...] = (0, 1, 2),
) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{split_name} missing columns: {sorted(missing)}")
    if frame["text"].isna().any():
        raise ValueError(f"{split_name} has null text")
    if not frame["label"].isin(valid_labels).all():
        raise ValueError(f"{split_name} has unexpected label values")
    return frame


def _load_split(
    csv_path: Path,
    split_name: str,
    label2id: dict[str, int],
    valid_labels: tuple[int, ...] = (0, 1, 2),
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame = map_labels(frame, label2id)
    return validate_schema(frame, split_name, valid_labels)


def load_splits(
    data_dir: Path,
    label2id: dict[str, int],
    valid_labels: tuple[int, ...] = (0, 1, 2),
) -> DatasetSplits:
    return DatasetSplits(
        train=_load_split(
            data_dir / "sicurre_train.csv", "train", label2id, valid_labels
        ),
        val=_load_split(data_dir / "sicurre_val.csv", "val", label2id, valid_labels),
        test=_load_split(data_dir / "sicurre_test.csv", "test", label2id, valid_labels),
    )


def summarize_split(frame: pd.DataFrame, id2label: dict[int, str]) -> dict[str, object]:
    counts = frame["label"].value_counts().sort_index()
    distribution = {id2label[index]: int(count) for index, count in counts.items()}
    return {"rows": int(len(frame)), "distribution": distribution}
