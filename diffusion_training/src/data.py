import itertools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset

from hat.splitter import HATSplitter
from src.utils import DataConfig, DataSliceConfig


TEXT_FIELDS = ("text", "content", "code", "markdown", "article", "body")
STACK_V2_DATASETS = {
    "bigcode/the-stack-v2",
    "bigcode/the-stack-v2-dedup",
    "bigcode/the-stack-v2-train-full-ids",
    "bigcode/the-stack-v2-train-smol-ids",
}


def normalize_probabilities(values: Sequence[float]) -> list[float]:
    total = float(sum(values))
    if total <= 0:
        raise ValueError("At least one dataset slice must have a positive proportion.")
    return [float(value) / total for value in values]


def extract_text(example: dict[str, Any], preferred_field: str | None = None) -> str:
    if preferred_field and preferred_field in example and example[preferred_field] is not None:
        return str(example[preferred_field])
    for field in TEXT_FIELDS:
        value = example.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _matches_language(example: dict[str, Any], language: str) -> bool:
    expected = language.lower()
    for field in ("lang", "language", "programming_language"):
        value = example.get(field)
        if value is not None and str(value).lower() == expected:
            return True
    for field in ("path", "max_stars_repo_path", "repo_name"):
        value = example.get(field)
        if value is not None and expected in str(value).lower():
            return True
    return False


def load_dataset_slice(
    name: str,
    slice_config: DataSliceConfig,
    split: str = "train",
):
    """Load one Hugging Face streaming slice.

    Access errors are intentionally allowed to surface to the caller so gated
    slices can be dropped and reweighted at the mixture boundary.
    """

    from datasets import load_dataset

    if slice_config.dataset in STACK_V2_DATASETS:
        raise ValueError(
            "Stack v2 rows contain Software Heritage blob IDs, not code text. "
            "Add an explicit S3 content-fetching loader before re-enabling this slice."
        )

    kwargs: dict[str, Any] = {
        "path": slice_config.dataset,
        "split": split,
        "streaming": True,
    }
    if slice_config.name:
        kwargs["name"] = slice_config.name

    dataset = load_dataset(**kwargs)
    if slice_config.filter_language:
        dataset = dataset.filter(lambda example: _matches_language(example, slice_config.filter_language))
    return dataset.map(lambda example: {"_slice": name, "_text": extract_text(example, slice_config.text_field)})


def build_mixed_stream(
    config: DataConfig,
    split: str = "train",
    drop_unavailable: bool = True,
    skip_examples_per_slice: int = 0,
):
    from datasets import interleave_datasets

    datasets = []
    probabilities = []
    names = []
    errors: dict[str, str] = {}

    for name, slice_config in config.slices.items():
        try:
            dataset = load_dataset_slice(name, slice_config, split=split)
            if skip_examples_per_slice > 0:
                dataset = dataset.skip(skip_examples_per_slice)
        except Exception as exc:
            if not drop_unavailable:
                raise
            errors[name] = repr(exc)
            continue
        datasets.append(dataset)
        probabilities.append(slice_config.proportion)
        names.append(name)

    if not datasets:
        raise RuntimeError(f"No dataset slices could be loaded. Errors: {errors}")

    mixed = interleave_datasets(
        datasets,
        probabilities=normalize_probabilities(probabilities),
        stopping_strategy="all_exhausted",
    )
    return mixed, names, normalize_probabilities(probabilities), errors


def synthetic_text_stream() -> Iterator[dict[str, str]]:
    examples = [
        ("english", "When was Rome founded? Rome was founded in 753 BC."),
        ("german", "Gruesse aus Koeln. Grüße aus Köln und Berlin."),
        ("math", "Let x+y=2. If x=1, then y=1. QED."),
        ("code", "def foo_bar(x):\n    return x**2 + 1\n"),
        ("web", "A small paragraph with punctuation, whitespace, and camelCaseWords."),
    ]
    while True:
        for name, text in examples:
            yield {"_slice": name, "_text": text}


class TextIterableDataset(IterableDataset):
    def __init__(self, examples: Iterable[dict[str, Any] | str]):
        self.examples = examples

    def __iter__(self):
        yield from self.examples


@dataclass
class EncodedText:
    text: str
    byte_ids: list[int]
    boundaries: torch.Tensor


class TextCollator:
    def __init__(
        self,
        sequence_length: int,
        special_token_dict: dict[str, int] | None = None,
        max_word_size: int = 100,
        pad_id: int = 0,
    ):
        self.sequence_length = sequence_length
        self.pad_id = pad_id
        self.splitter = HATSplitter(special_token_dict=special_token_dict or {}, max_word_size=max_word_size)

    def encode_text(self, text: str) -> EncodedText:
        words = self.splitter.encode(text)
        truncated_words: list[list[int]] = []
        remaining = self.sequence_length
        for word in words:
            if remaining <= 0:
                break
            if len(word) <= remaining:
                truncated_words.append(list(word))
                remaining -= len(word)
            else:
                break

        byte_ids = list(itertools.chain.from_iterable(truncated_words))
        if not byte_ids:
            byte_ids = [self.pad_id]
            truncated_words = [[self.pad_id]]

        boundaries = torch.tensor(
            [0] + list(itertools.accumulate(len(word) for word in truncated_words if len(word) > 0)),
            dtype=torch.int32,
        )
        return EncodedText(text=text, byte_ids=byte_ids, boundaries=boundaries)

    def decode_bytes(self, byte_ids: Sequence[int]) -> str:
        return self.splitter.decode(list(byte_ids), errors="strict")

    def __call__(self, examples: Sequence[dict[str, Any] | str]) -> dict[str, Any]:
        encoded: list[EncodedText] = []
        for example in examples:
            if isinstance(example, str):
                text = example
            else:
                text = str(example.get("_text") or extract_text(example))
            encoded.append(self.encode_text(text))

        batch_size = len(encoded)
        byte_ids = torch.full((batch_size, self.sequence_length), self.pad_id, dtype=torch.long)
        labels = torch.full((batch_size, self.sequence_length), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, self.sequence_length), dtype=torch.bool)
        lengths = torch.zeros(batch_size, dtype=torch.long)

        for index, item in enumerate(encoded):
            length = min(len(item.byte_ids), self.sequence_length)
            values = torch.tensor(item.byte_ids[:length], dtype=torch.long)
            byte_ids[index, :length] = values
            attention_mask[index, :length] = True
            lengths[index] = length
            if length > 1:
                labels[index, : length - 1] = values[1:]

        boundaries = [item.boundaries for item in encoded]
        return {
            "byte_ids": byte_ids,
            "input_ids": byte_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "lengths": lengths,
            "word_boundaries": boundaries,
            "cumulative_seq_lengths_per_word": boundaries,
            "texts": [item.text for item in encoded],
        }


def make_dataloader(
    examples: Iterable[dict[str, Any] | str],
    collator: TextCollator,
    batch_size: int,
) -> DataLoader:
    return DataLoader(TextIterableDataset(examples), batch_size=batch_size, collate_fn=collator)


def write_jsonl_examples(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
