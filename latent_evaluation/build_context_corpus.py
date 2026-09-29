"""One-off generator for latent_evaluation/data/context_corpus.txt.

The corpus feeds the in-context covariance analysis: 500 Wikipedia documents,
each exactly 350 whitespace-separated words, one document per line. The frozen
output file is the reference artifact — this script only documents how it was
drawn and never needs to run again.

Provenance:
- Source: wikimedia/wikipedia, config 20231101.en — the same Wikipedia snapshot
  the diffusion training mixture streams (see configs/training/*, slices
  wikipedia_en) — read in dump order via streaming.
- Pool: the first POOL_SIZE articles whose whitespace word count >= DOC_WORDS.
- Sample: random.Random(SEED).sample(pool, N_DOCS).
- Storage: " ".join(text.split()[:DOC_WORDS]) — truncation counts whitespace
  words and rejoins with single spaces, so a document can never contain a
  newline and the file needs no escaping or metadata.

Run once (diff-train env, needs network):
    /home/fwichert/miniconda3/envs/diff-train/bin/python latent_evaluation/build_context_corpus.py
"""

import random
from pathlib import Path

from datasets import load_dataset

N_DOCS = 500
DOC_WORDS = 350
POOL_SIZE = 5000
SEED = 42
OUT_PATH = Path(__file__).resolve().parent / "data" / "context_corpus.txt"


def main() -> None:
    stream = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    pool = []
    scanned = 0
    for row in stream:
        scanned += 1
        words = row["text"].split()
        if len(words) >= DOC_WORDS:
            pool.append(" ".join(words[:DOC_WORDS]))
            if len(pool) >= POOL_SIZE:
                break
    if len(pool) < N_DOCS:
        raise RuntimeError(f"only {len(pool)} qualifying articles found, need {N_DOCS}")

    docs = random.Random(SEED).sample(pool, N_DOCS)
    OUT_PATH.write_text("\n".join(docs) + "\n", encoding="utf-8")
    print(f"scanned {scanned} articles, pool {len(pool)}, wrote {len(docs)} docs -> {OUT_PATH}")


if __name__ == "__main__":
    main()
