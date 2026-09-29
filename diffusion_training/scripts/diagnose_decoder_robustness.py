import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_decode_ceiling import per_word_exact, norm_edit
from scripts.text_metrics import DROP_FIRST_N_WORDS
from src.diffusion_train import load_frozen_autoencoder
from src.sampling import decode_latents, encode_prompt, resolve_latent_stats
from src.train import _dtype_from_name, build_training_loader
from src.utils import load_diffusion_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Diffusion config: supplies data mixture, stats path, model dims.")
    parser.add_argument(
        "--ae", action="append", required=True, metavar="LABEL=CHECKPOINT",
        help="Autoencoder to evaluate (repeatable), e.g. baseline=checkpoints/.../step_040000.pt",
    )
    parser.add_argument("--sigmas", default="0,0.1,0.2,0.3,0.4,0.5",
                        help="Comma-separated noise levels in STANDARDIZED units.")
    parser.add_argument("--num-docs", type=int, default=25)
    parser.add_argument("--max-words", type=int, default=48)
    parser.add_argument("--bytes-per-word", type=int, default=6)
    parser.add_argument("--seed-byte", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0, help="Base seed for the noise draws.")
    parser.add_argument("--results-dir", default="notes/roadmap-artifacts")
    parser.add_argument("--device")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_diffusion_config(args.config)
    device = torch.device(args.device or config.training.device)
    dtype = _dtype_from_name(config.training.dtype)
    sigmas = [float(s) for s in args.sigmas.split(",")]
    checkpoints = dict(spec.split("=", 1) for spec in args.ae)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Collect the doc texts once (same for every AE).
    loader, _ = build_training_loader(config)
    texts: list[str] = []
    for batch in loader:
        texts.extend(batch["texts"])
        if len(texts) >= args.num_docs:
            break
    texts = texts[: args.num_docs]

    # scores[label][sigma] = (exact_num, exact_den, edit_sum, n_docs)
    scores: dict[str, dict[float, list[float]]] = {
        label: {s: [0, 0, 0.0, 0] for s in sigmas} for label in checkpoints
    }

    for label, ckpt_path in checkpoints.items():
        config.autoencoder.checkpoint = ckpt_path
        autoencoder = load_frozen_autoencoder(config, device)
        latent_stats = resolve_latent_stats({}, config, device)
        if latent_stats is None:
            raise SystemExit("latent stats required (per-dim std scales the noise); set autoencoder.latent_stats_path")
        std = latent_stats["std"].to(device=device).flatten()

        doc_index = 0
        for text in texts:
            prompt = encode_prompt(autoencoder, text, config.data.sequence_length, device)
            n_words = min(len(prompt.word_lengths), args.max_words)
            if n_words < DROP_FIRST_N_WORDS + 2:
                continue
            true_lengths = [int(x) for x in prompt.word_lengths[:n_words]]
            n_true_bytes = sum(true_lengths)
            source_text = bytes(prompt.byte_ids[:n_true_bytes]).decode("utf-8", errors="replace")
            z_raw = prompt.z_words[:, :n_words, :].to(dtype=torch.float32)

            for s_index, sigma in enumerate(sigmas):
                # Same eps for every AE: seeded by (doc, sigma), independent of label.
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + 100_003 * doc_index + 1_009 * s_index
                )
                eps = torch.randn(z_raw.shape, generator=generator, device=device, dtype=torch.float32)
                z_noisy = (z_raw + sigma * std * eps).to(dtype=dtype)
                decoded = decode_latents(
                    autoencoder, z_noisy, num_bytes=n_true_bytes, bytes_per_word=args.bytes_per_word,
                    seed_byte=args.seed_byte, device=device, word_lengths=true_lengths,
                )
                num, den = per_word_exact(source_text, decoded)
                cell = scores[label][sigma]
                cell[0] += num
                cell[1] += den
                cell[2] += norm_edit(source_text, decoded)
                cell[3] += 1
            doc_index += 1
        print(f"done: {label} ({doc_index} docs)", flush=True)
        del autoencoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    labels = list(checkpoints)
    lines = [f"{'sigma':>6} " + " ".join(f"{lab + ':exact':>16} {lab + ':edit':>12}" for lab in labels)]
    for sigma in sigmas:
        row = f"{sigma:>6.2f} "
        for lab in labels:
            num, den, edit, n = scores[lab][sigma]
            row += f" {num / max(1, den):>16.4f} {edit / max(1, n):>12.4f}"
        lines.append(row)
    report = "\n".join(lines)
    print("\nword-exact (difflib alignment, first 2 words dropped) and normalized edit distance,")
    print("free-run decode with TRUE word lengths (condition-B path), noise in standardized units:\n")
    print(report)

    out = results_dir / "decoder-robustness-curve"
    out.with_suffix(".txt").write_text(report + "\n")
    out.with_suffix(".json").write_text(json.dumps({
        "config": args.config, "checkpoints": checkpoints, "sigmas": sigmas,
        "num_docs": args.num_docs, "max_words": args.max_words, "seed": args.seed,
        "scores": {lab: {str(s): scores[lab][s] for s in sigmas} for lab in labels},
    }, indent=1))
    print(f"\nwrote {out}.txt / .json")


if __name__ == "__main__":
    main()
