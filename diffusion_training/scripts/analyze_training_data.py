import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import TextCollator, build_mixed_stream
from src.utils import load_diffusion_config, resolve_config_path

# Docs are clipped to this many chars before word-splitting: the byte cap keeps at
# most sequence_length bytes and a char is >= 1 byte, so kept-word stats are exact.
# Raw word counts for clipped docs are extrapolated from the clipped prefix.
CLIP_CHARS_FACTOR = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Diffusion training config (path or bare name in configs/training/)")
    parser.add_argument("--sample-examples", type=int, default=5000)
    parser.add_argument("--output-dir", default="artifacts/data_analysis")
    parser.add_argument(
        "--no-skip-holdout",
        action="store_true",
        help="Sample from the head of each slice instead of skipping the validation holdout like the trainer does.",
    )
    parser.add_argument(
        "--no-size-check",
        action="store_true",
        help="Skip fetching dataset metadata from the Hub for the epoch/budget check.",
    )
    parser.add_argument(
        "--normal-exit",
        action="store_true",
        help="Use normal Python shutdown instead of bypassing native extension cleanup.",
    )
    return parser.parse_args()


def percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)

    def pct(p: float) -> float:
        return float(ordered[min(n - 1, int(p * n))])

    return {
        "mean": sum(ordered) / n,
        "p05": pct(0.05),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": float(ordered[-1]),
    }


def fetch_slice_sizes(config, drop_unavailable: bool) -> dict[str, int | None]:
    from datasets import load_dataset_builder

    sizes: dict[str, int | None] = {}
    for name, slice_config in config.data.slices.items():
        try:
            builder = load_dataset_builder(slice_config.dataset, name=slice_config.name)
            split_info = (builder.info.splits or {}).get("train")
            sizes[name] = int(split_info.num_examples) if split_info else None
        except Exception as exc:
            if not drop_unavailable:
                raise
            print(f"[size-check] {name}: metadata unavailable ({exc!r})", file=sys.stderr)
            sizes[name] = None
    return sizes


def collect_sample(config, sample_examples: int, skip_holdout: bool) -> dict[str, Any]:
    sequence_length = config.data.sequence_length
    max_words = config.diffusion.max_words
    clip_chars = sequence_length * CLIP_CHARS_FACTOR
    splitter = TextCollator(sequence_length=sequence_length).splitter

    skip = 0
    if skip_holdout:
        skip = (
            config.training.validation_holdout_examples_per_slice
            + config.training.train_skip_examples_per_slice
        )
    stream, names, probabilities, errors = build_mixed_stream(
        config.data,
        split="train",
        drop_unavailable=config.training.drop_unavailable_data,
        skip_examples_per_slice=skip,
    )

    per_slice: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "empty": 0,
            "duplicates": 0,
            "byte_truncated": 0,
            "over_max_words": 0,
            "raw_bytes": [],
            "raw_words_est": [],
            "kept_words": [],
            "kept_bytes_total": 0,
            "raw_bytes_total": 0,
            "latent_words_total": 0,
            "hashes": set(),
        }
    )
    bytes_per_word = Counter()

    for index, example in enumerate(stream):
        if index >= sample_examples:
            break
        slice_name = example.get("_slice", "unknown")
        text = example.get("_text") or ""
        stats = per_slice[slice_name]
        stats["count"] += 1

        if not text.strip():
            stats["empty"] += 1
            continue

        digest = hashlib.md5(text.encode("utf-8")).digest()
        if digest in stats["hashes"]:
            stats["duplicates"] += 1
        else:
            stats["hashes"].add(digest)

        raw_bytes = len(text.encode("utf-8"))
        clipped = len(text) > clip_chars
        words = splitter.encode(text[:clip_chars] if clipped else text)
        word_lens = [len(word) for word in words]
        split_bytes = sum(word_lens)

        # Mirror TextCollator.encode_text: whole words until the byte budget is hit.
        kept_words = 0
        kept_bytes = 0
        remaining = sequence_length
        for length in word_lens:
            if remaining <= 0 or length > remaining:
                break
            kept_words += 1
            kept_bytes += length
            remaining -= length
            bytes_per_word[length] += 1

        if clipped and split_bytes > 0:
            raw_words = int(round(len(word_lens) * raw_bytes / split_bytes))
        else:
            raw_words = len(word_lens)

        stats["raw_bytes"].append(raw_bytes)
        stats["raw_words_est"].append(raw_words)
        stats["kept_words"].append(kept_words)
        stats["raw_bytes_total"] += raw_bytes
        stats["kept_bytes_total"] += kept_bytes
        stats["latent_words_total"] += min(kept_words, max_words)
        if raw_bytes > kept_bytes:
            stats["byte_truncated"] += 1
        if kept_words > max_words:
            stats["over_max_words"] += 1

        if (index + 1) % 1000 == 0:
            print(f"[sample] {index + 1}/{sample_examples} examples", file=sys.stderr)

    for stats in per_slice.values():
        del stats["hashes"]
    return {
        "per_slice": dict(per_slice),
        "bytes_per_word": bytes_per_word,
        "loaded_slices": names,
        "probabilities": probabilities,
        "errors": errors,
    }


def summarize(config, sample: dict[str, Any], slice_sizes: dict[str, int | None], skip: int) -> dict[str, Any]:
    training = config.training
    budget = training.max_steps * training.batch_size * training.gradient_accumulation_steps
    max_words = config.diffusion.max_words

    slices_summary = {}
    for name, stats in sample["per_slice"].items():
        count = stats["count"]
        analyzed = max(1, count - stats["empty"])
        kept_words_total = sum(stats["kept_words"])
        probability = sample["probabilities"][sample["loaded_slices"].index(name)]
        expected_examples = budget * probability
        size = slice_sizes.get(name)
        epochs = expected_examples / max(1, size - skip) if size else None
        slices_summary[name] = {
            "sampled_examples": count,
            "realized_proportion": count / max(1, sum(s["count"] for s in sample["per_slice"].values())),
            "configured_proportion": probability,
            "empty_rate": stats["empty"] / max(1, count),
            "duplicate_rate": stats["duplicates"] / max(1, analyzed),
            "raw_doc_bytes": percentiles(stats["raw_bytes"]),
            "raw_doc_words_est": percentiles(stats["raw_words_est"]),
            "kept_words": percentiles(stats["kept_words"]),
            "pct_docs_byte_truncated": stats["byte_truncated"] / max(1, analyzed),
            "pct_bytes_dropped_by_cap": 1.0 - stats["kept_bytes_total"] / max(1, stats["raw_bytes_total"]),
            "pct_docs_over_max_words": stats["over_max_words"] / max(1, analyzed),
            "pct_latents_dropped_by_max_words": 1.0 - stats["latent_words_total"] / max(1, kept_words_total),
            "advertised_train_examples": size,
            "expected_examples_for_run": expected_examples,
            "implied_epochs": epochs,
        }

    return {
        "config": str(resolve_config_path_str(config)),
        "budget_examples": budget,
        "budget_formula": f"{training.max_steps} steps x {training.batch_size} batch x "
        f"{training.gradient_accumulation_steps} grad_accum",
        "sequence_length_bytes": config.data.sequence_length,
        "diffusion_max_words": max_words,
        "holdout_skip_per_slice": skip,
        "mixture_errors": sample["errors"],
        "slices": slices_summary,
    }


def resolve_config_path_str(config) -> str:
    return getattr(config, "_source_path", "<config>")


def plot_distributions(sample: dict[str, Any], config, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    surface = "#fcfcfb"
    ink = "#0b0b0b"
    ink_2 = "#52514e"
    grid = "#e4e3df"
    series = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]

    slices = sorted(sample["per_slice"].keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=surface)

    def style(ax, title, xlabel):
        ax.set_facecolor(surface)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(grid)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=ink_2, labelsize=9)
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.set_xlabel(xlabel, color=ink_2, fontsize=9)
        ax.set_ylabel("fraction of slice docs", color=ink_2, fontsize=9)

    max_words = config.diffusion.max_words
    sequence_length = config.data.sequence_length

    ax = axes[0]
    upper = max(max(sample["per_slice"][name]["kept_words"] or [1]) for name in slices)
    bins = np.linspace(0, upper + 1, 60)
    for i, name in enumerate(slices):
        values = sample["per_slice"][name]["kept_words"]
        ax.hist(values, bins=bins, histtype="step", linewidth=2, color=series[i % len(series)],
                weights=np.full(len(values), 1.0 / max(1, len(values))), label=name)
    ax.axvline(max_words, color=ink_2, linewidth=1, linestyle=":")
    ax.text(max_words, ax.get_ylim()[1] * 0.95, f" max_words={max_words}", color=ink_2, fontsize=8, va="top")
    style(ax, "Words per document (after byte cap)", "words (diffusion sequence length)")
    ax.legend(frameon=False, fontsize=9, labelcolor=ink)

    ax = axes[1]
    all_raw = [v for name in slices for v in sample["per_slice"][name]["raw_bytes"]]
    log_bins = np.geomspace(max(1, min(all_raw)), max(all_raw), 60)
    for i, name in enumerate(slices):
        values = sample["per_slice"][name]["raw_bytes"]
        ax.hist(values, bins=log_bins, histtype="step", linewidth=2, color=series[i % len(series)],
                weights=np.full(len(values), 1.0 / max(1, len(values))), label=name)
    ax.set_xscale("log")
    ax.axvline(sequence_length, color=ink_2, linewidth=1, linestyle=":")
    ax.text(sequence_length, ax.get_ylim()[1] * 0.95, f" cap={sequence_length}B", color=ink_2, fontsize=8, va="top")
    style(ax, "Raw document size", "UTF-8 bytes (log scale)")
    ax.legend(frameon=False, fontsize=9, labelcolor=ink)

    ax = axes[2]
    counter = sample["bytes_per_word"]
    total = sum(counter.values())
    lengths = np.arange(1, max(counter) + 1)
    fractions = np.array([counter.get(int(l), 0) / total for l in lengths])
    ax.bar(lengths, fractions, width=1.0, color=series[0], edgecolor=surface, linewidth=0.4)
    style(ax, "Bytes per word (kept words, mixture)", "bytes")
    ax.set_ylabel("fraction of words", color=ink_2, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=surface)
    plt.close(fig)


def format_report(summary: dict[str, Any]) -> str:
    lines = [
        f"config: {summary['config']}",
        f"data budget: {summary['budget_examples']:,} examples ({summary['budget_formula']})",
        f"byte cap: {summary['sequence_length_bytes']}  |  diffusion max_words: {summary['diffusion_max_words']}"
        f"  |  holdout skip/slice: {summary['holdout_skip_per_slice']:,}",
    ]
    if summary["mixture_errors"]:
        lines.append(f"DROPPED SLICES: {summary['mixture_errors']}")
    for name, s in summary["slices"].items():
        epochs = s["implied_epochs"]
        lines += [
            "",
            f"[{name}] sampled={s['sampled_examples']}  mix realized={s['realized_proportion']:.3f}"
            f" configured={s['configured_proportion']:.3f}",
            f"  raw doc bytes: p50={s['raw_doc_bytes'].get('p50', 0):,.0f}"
            f" p95={s['raw_doc_bytes'].get('p95', 0):,.0f} max={s['raw_doc_bytes'].get('max', 0):,.0f}",
            f"  words/doc after byte cap: p50={s['kept_words'].get('p50', 0):.0f}"
            f" p95={s['kept_words'].get('p95', 0):.0f} max={s['kept_words'].get('max', 0):.0f}",
            f"  byte-cap truncation: {s['pct_docs_byte_truncated']:.1%} of docs,"
            f" {s['pct_bytes_dropped_by_cap']:.1%} of bytes dropped",
            f"  max_words truncation: {s['pct_docs_over_max_words']:.1%} of docs over cap,"
            f" {s['pct_latents_dropped_by_max_words']:.1%} of word latents dropped",
            f"  empty: {s['empty_rate']:.2%}  duplicates: {s['duplicate_rate']:.2%}",
            f"  budget: needs {s['expected_examples_for_run']:,.0f} examples; slice has "
            f"{s['advertised_train_examples']:,} -> {epochs:.2f} epochs"
            if s["advertised_train_examples"]
            else f"  budget: needs {s['expected_examples_for_run']:,.0f} examples; slice size unknown",
        ]
        if epochs is not None and epochs > 1.0:
            lines.append(f"  WARNING: run consumes the {name} slice {epochs:.2f}x (data will repeat).")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    config = load_diffusion_config(config_path)
    config._source_path = str(config_path)

    skip = 0
    if not args.no_skip_holdout:
        skip = (
            config.training.validation_holdout_examples_per_slice
            + config.training.train_skip_examples_per_slice
        )

    slice_sizes: dict[str, int | None] = {name: None for name in config.data.slices}
    if not args.no_size_check:
        slice_sizes = fetch_slice_sizes(config, config.training.drop_unavailable_data)

    sample = collect_sample(config, args.sample_examples, skip_holdout=not args.no_skip_holdout)
    summary = summarize(config, sample, slice_sizes, skip)

    output_dir = Path(args.output_dir) / config_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "distributions.png"
    plot_distributions(sample, config, plot_path)

    report = format_report(summary)
    (output_dir / "summary.txt").write_text(report + "\n", encoding="utf-8")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(report)
    print(f"\nplot={plot_path}")
    print(f"summary={output_dir / 'summary.txt'}")
    print(f"json={output_dir / 'summary.json'}")
    if not args.normal_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
