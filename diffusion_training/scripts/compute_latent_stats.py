import argparse
import datetime
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diffusion_train import encode_latent_batch, load_frozen_autoencoder
from src.train import _dtype_from_name, build_training_loader, move_batch_to_device
from src.utils import load_diffusion_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Diffusion YAML config (data mixture + autoencoder checkpoint).")
    parser.add_argument("--output", default="artifacts/latents/latent_stats.pt")
    parser.add_argument("--num-sequences", type=int, default=1024)
    parser.add_argument("--holdout-batches", type=int, default=4, help="Extra batches for the sanity check.")
    parser.add_argument("--device")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_diffusion_config(args.config)
    device = torch.device(args.device or config.training.device)
    dtype = _dtype_from_name(config.training.dtype)
    latent_dim = config.diffusion.latent_dim

    autoencoder = load_frozen_autoencoder(config, device)
    loader, data_info = build_training_loader(config)

    total = torch.zeros(latent_dim, device=device, dtype=torch.float64)
    total_sq = torch.zeros(latent_dim, device=device, dtype=torch.float64)
    word_count = 0
    sequence_count = 0
    holdout: list[torch.Tensor] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        latent_batch = encode_latent_batch(
            autoencoder,
            batch,
            max_words=config.diffusion.max_words,
            latent_dim=latent_dim,
            dtype=dtype,
            device=device,
        )
        valid = latent_batch["z"][latent_batch["mask"]].to(torch.float64)
        if sequence_count < args.num_sequences:
            total += valid.sum(dim=0)
            total_sq += valid.pow(2).sum(dim=0)
            word_count += valid.size(0)
            sequence_count += int(latent_batch["mask"].size(0))
            if sequence_count % 128 < latent_batch["mask"].size(0):
                print(f"sequences={sequence_count} words={word_count}", flush=True)
        else:
            holdout.append(valid.float())
            if len(holdout) >= args.holdout_batches:
                break

    mean = (total / max(1, word_count)).float()
    variance = total_sq / max(1, word_count) - (total / max(1, word_count)).pow(2)
    std = variance.clamp_min(0).sqrt().float()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": mean.cpu(),
            "std": std.cpu(),
            "num_words": word_count,
            "num_sequences": sequence_count,
            "autoencoder_checkpoint": config.autoencoder.checkpoint,
            "config": args.config,
            "data": data_info,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        output_path,
    )

    raw_norm = None
    print(f"output={output_path}")
    print(f"words={word_count} sequences={sequence_count}")
    print(f"raw per-dim std: mean={std.mean():.4f} min={std.min():.4f} max={std.max():.4f}")
    print(f"raw per-dim mean absmax={mean.abs().max():.4f}")
    if holdout:
        held = torch.cat(holdout, dim=0)
        raw_norm = held.norm(dim=-1).mean()
        standardized = (held - mean) / std.clamp_min(1e-3)
        print(f"holdout raw: elem std={held.std():.4f} norm mean={raw_norm:.3f}")
        print(
            f"holdout standardized: elem std={standardized.std():.4f} "
            f"norm mean={standardized.norm(dim=-1).mean():.3f} "
            f"(targets ~1 and ~{latent_dim ** 0.5:.1f})"
        )


if __name__ == "__main__":
    main()
