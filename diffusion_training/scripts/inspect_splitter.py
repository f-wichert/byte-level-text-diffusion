import contextlib
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hat.splitter import HATSplitter


EXAMPLES = [
    " Hello, world!",
    "camelCase foo_bar x+y=2",
    "Gruesse aus Koeln. Grüße aus Köln.",
    "def fooBar(x): return x**2 + 1",
]


def inspect() -> None:
    splitter = HATSplitter(special_token_dict={}, max_word_size=100)
    for text in EXAMPLES:
        words = splitter.encode(text)
        flat = list(itertools.chain.from_iterable(words))
        cumulative = torch.tensor(
            [0] + list(itertools.accumulate(len(word) for word in words if len(word) > 0)),
            dtype=torch.int32,
        )
        round_trip = splitter.decode(flat, errors="strict")
        print("text:", repr(text))
        print("words:", words)
        print("word byte lengths:", [len(word) for word in words])
        print("flat byte length:", len(flat))
        print("cumulative_seq_lengths_per_word:", cumulative, cumulative.shape, cumulative.dtype)
        print("round_trip_ok:", round_trip == text)
        print()


def main() -> None:
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            with contextlib.redirect_stdout(handle):
                inspect()
        print(f"Wrote splitter inspection to {output_path}")
    else:
        inspect()


if __name__ == "__main__":
    main()
