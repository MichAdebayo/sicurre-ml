from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.tokenizer import build_dataset_dict, load_tokenizer, tokenize_batch


def test_build_dataset_dict_creates_correct_splits() -> None:
    train_df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    val_df = pd.DataFrame({"text": ["c"], "label": [2]})
    test_df = pd.DataFrame({"text": ["d"], "label": [0]})
    ds = build_dataset_dict(train_df, val_df, test_df)
    assert len(ds["train"]) == 2
    assert len(ds["val"]) == 1
    assert len(ds["test"]) == 1


def test_build_dataset_dict_preserves_columns() -> None:
    df = pd.DataFrame({"text": ["hello"], "label": [0]})
    ds = build_dataset_dict(df, df, df)
    assert "text" in ds["train"].column_names
    assert "label" in ds["train"].column_names


def test_load_tokenizer_calls_auto_tokenizer() -> None:
    mock_tok = MagicMock()
    with patch(
        "src.data.tokenizer.AutoTokenizer.from_pretrained", return_value=mock_tok
    ):
        result = load_tokenizer("some-model", token="tok")
    assert result is mock_tok


def test_tokenize_batch_calls_tokenizer_with_correct_args() -> None:
    mock_tokenizer = MagicMock(
        return_value={"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}
    )
    batch = {"text": ["hello world"]}
    result = tokenize_batch(batch, mock_tokenizer, max_length=128)
    mock_tokenizer.assert_called_once_with(
        ["hello world"], truncation=True, max_length=128, padding="max_length"
    )
    assert "input_ids" in result
