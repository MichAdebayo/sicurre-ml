from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset, DatasetDict
from pandas import DataFrame
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass(slots=True)
class TokenizedSplits:
    tokenizer: PreTrainedTokenizerBase
    train: Dataset
    val: Dataset
    test: Dataset


def build_dataset_dict(
    train_df: DataFrame, val_df: DataFrame, test_df: DataFrame
) -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_pandas(
                train_df[["text", "label"]], preserve_index=False
            ),
            "val": Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False),
            "test": Dataset.from_pandas(
                test_df[["text", "label"]], preserve_index=False
            ),
        }
    )


def load_tokenizer(
    model_name: str, token: str | None = None
) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name, token=token)


def tokenize_batch(
    batch: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, list[int]]:
    return tokenizer(
        batch["text"], truncation=True, max_length=max_length, padding="max_length"
    )


def prepare_tokenized_splits(
    train_df: DataFrame,
    val_df: DataFrame,
    test_df: DataFrame,
    model_name: str,
    max_length: int,
    hf_token: str | None = None,
    batch_size: int = 256,
) -> TokenizedSplits:
    dataset_dict = build_dataset_dict(train_df, val_df, test_df)
    tokenizer = load_tokenizer(model_name, token=hf_token)

    tokenized = dataset_dict.map(
        lambda batch: tokenize_batch(batch, tokenizer=tokenizer, max_length=max_length),
        batched=True,
        batch_size=batch_size,
    )
    for split_name in tokenized.keys():
        tokenized[split_name].set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )

    return TokenizedSplits(
        tokenizer=tokenizer,
        train=tokenized["train"],
        val=tokenized["val"],
        test=tokenized["test"],
    )
