import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sample_diffusion import decode_latents
from src.diffusion_model import LatentFlowTransformer, build_diffusion_backbone
from src.diffusion_train import (
    encode_latent_batch,
    load_frozen_autoencoder,
    load_latent_stats,
    path_coefficients,
    velocity_to_x0,
    x0_to_velocity,
)
from src.train import _dtype_from_name, build_training_loader, move_batch_to_device
from src.utils import load_checkpoint, load_diffusion_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to diffusion YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Diffusion checkpoint to sample from.")
    parser.add_argument(
        "--t-values",
        default="0.0,0.25,0.5,0.75,0.9",
        help="Comma-separated start times to sweep (0 = pure noise / unconditional, "
        "near 1 = little noise / reconstruction).",
    )
    parser.add_argument("--num-samples", type=int, default=4, help="Real sequences to seed from.")
    parser.add_argument(
        "--use-overfit-targets",
        action="store_true",
        help="Seed from the exact sequence(s) the model was overfit on (stored in the checkpoint by "
        "overfit_diffusion.py --save-path) instead of a fresh random batch.",
    )
    parser.add_argument("--sampling-steps", type=int, help="Euler steps over the [t, 1] window.")
    parser.add_argument("--num-bytes", type=int)
    parser.add_argument("--bytes-per-word", type=int, default=6)
    parser.add_argument(
        "--length-predictor",
        help="Length-predictor artifact (scripts/train_length_predictor.py); replaces the fixed "
        "bytes-per-word grid when true word_lengths are unavailable (e.g. fresh-batch mode).",
    )
    parser.add_argument("--seed-byte", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def teacher_forced_sample(
    model: LatentFlowTransformer,
    z_data: torch.Tensor,
    mask: torch.Tensor,
    t_start: float,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Noise ``z_data`` (standardized working space) to ``t_start`` and integrate to 1."""
    path = getattr(model.config, "path", "linear")
    self_cond = getattr(model.config, "self_conditioning", False)
    prediction_mode = getattr(model.config, "prediction", "velocity")

    eps = torch.randn_like(z_data)
    alpha, sigma, _, _ = path_coefficients(t_start, path)
    z = alpha * z_data + sigma * eps

    num_samples = z.size(0)
    z_self = None
    dt = (1.0 - t_start) / steps
    for index in range(steps):
        t_value = t_start + (index + 0.5) * dt
        t = torch.full((num_samples,), t_value, device=device)
        raw = model(z, t, mask, z_self)
        if prediction_mode == "x0":
            x0_hat = raw
            velocity = x0_to_velocity(z, x0_hat, t, path)
        else:
            velocity = raw
            x0_hat = velocity_to_x0(z, raw, t, path) if self_cond else None
        if self_cond:
            z_self = x0_hat.to(dtype)
        z = z + dt * velocity
    return z


def recon_metrics(z_gen: torch.Tensor, z_data: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """Masked per-position MSE and mean cosine similarity vs the source latent."""
    valid = mask.unsqueeze(-1)
    gen = z_gen.float()
    ref = z_data.float()
    mse = (((gen - ref) ** 2) * valid).sum() / valid.sum().clamp_min(1.0) / z_data.size(-1)
    cos = torch.cosine_similarity(gen, ref, dim=-1)
    cos = (cos * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
    return float(mse.item()), float(cos.item())


def decode_text(autoencoder, z_std, mask_row, latent_stats, args, dtype, device, word_lengths=None) -> str:
    word_count = int(mask_row.sum().item())
    if word_count <= 0:
        return "<empty>"
    z = z_std[:, :word_count, :].float()
    if latent_stats is not None:
        z = z * latent_stats["std"] + latent_stats["mean"]
    z = z.to(dtype=dtype)
    # Prefer real/predicted per-word byte boundaries over the fixed bytes-per-word grid, which
    # otherwise drifts out of alignment with variable-length words (see README "Known problems").
    lengths = word_lengths[:word_count] if word_lengths is not None else None
    num_bytes = args.num_bytes or (sum(lengths) if lengths is not None else word_count * args.bytes_per_word)
    return decode_latents(
        autoencoder,
        z,
        num_bytes=num_bytes,
        bytes_per_word=args.bytes_per_word,
        seed_byte=args.seed_byte,
        device=device,
        word_lengths=lengths,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    config = load_diffusion_config(args.config)
    if args.allow_download:
        config.autoencoder.local_files_only = False

    device = torch.device(args.device or config.training.device)
    dtype = _dtype_from_name(config.training.dtype)

    autoencoder = load_frozen_autoencoder(config, device)
    model = build_diffusion_backbone(config.diffusion).to(device=device, dtype=dtype)
    checkpoint = load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    latent_stats = checkpoint.get("extra", {}).get("latent_stats")
    if latent_stats is None and config.autoencoder.latent_stats_path:
        latent_stats = load_latent_stats(config.autoencoder.latent_stats_path, device, config.diffusion.latent_dim)
    if latent_stats is None:
        print(
            "WARNING: no latent stats in checkpoint or config; operating on raw latents.",
            file=sys.stderr,
        )
    else:
        latent_stats = {
            "mean": latent_stats["mean"].to(device=device, dtype=torch.float32),
            "std": latent_stats["std"].to(device=device, dtype=torch.float32),
        }

    length_predictor = None
    if args.length_predictor:
        from scripts.train_length_predictor import LengthPredictor

        length_predictor = LengthPredictor.load(args.length_predictor, device)

    # Seed latents (standardized working space): either the exact overfit sequence(s) embedded in
    # the checkpoint, or a fresh batch of real data encoded on the fly.
    sources: list[str] | None = None
    word_lengths_all: list[list[int]] | None = None  # true per-word byte lengths, when available
    if args.use_overfit_targets:
        overfit = checkpoint.get("extra", {}).get("overfit")
        if overfit is None:
            raise SystemExit(
                "Checkpoint has no 'overfit' targets. Re-run overfit_diffusion.py with --save-path, "
                "or drop --use-overfit-targets to seed from a fresh batch."
            )
        z_data = overfit["z_data"].to(device=device, dtype=dtype)
        mask = overfit["mask"].to(device=device, dtype=torch.bool)
        sources = overfit.get("sources")
        word_lengths_all = overfit.get("word_lengths")
        take = min(args.num_samples, z_data.size(0))
        z_data = z_data[:take]
        mask = mask[:take]
    else:
        loader, _ = build_training_loader(config)
        batch = move_batch_to_device(next(iter(loader)), device)
        latent_batch = encode_latent_batch(
            autoencoder,
            batch,
            max_words=config.diffusion.max_words,
            latent_dim=config.diffusion.latent_dim,
            dtype=dtype,
            device=device,
            latent_stats=latent_stats,
        )
        take = min(args.num_samples, latent_batch["z"].size(0))
        z_data = latent_batch["z"][:take]
        mask = latent_batch["mask"][:take]

    steps = args.sampling_steps or config.diffusion.sampling_steps
    t_values = [float(v) for v in args.t_values.split(",") if v.strip()]

    print(f"checkpoint={args.checkpoint} step={checkpoint.get('step')}")
    print(f"path={getattr(model.config, 'path', 'linear')} steps={steps} "
          f"standardized={latent_stats is not None} num_samples={take} "
          f"seed={'overfit-targets' if args.use_overfit_targets else 'fresh-batch'}")
    grid = "true-word-lengths" if word_lengths_all is not None else (
        "length-predictor" if length_predictor is not None else f"fixed-{args.bytes_per_word}byte-grid"
    )
    print(f"decode boundaries: {grid}")
    print("note: the first 1-2 words are unreliable (decode is seeded with a space byte)")

    def resolve_lengths(i: int, z_std_row: torch.Tensor, word_count: int) -> list[int] | None:
        """Per-word byte lengths for decoding: true (stored) > predicted > fixed grid (None)."""
        if word_lengths_all is not None and i < len(word_lengths_all):
            return word_lengths_all[i]
        if length_predictor is not None:
            return length_predictor.predict_lengths(z_std_row[:word_count]).tolist()
        return None

    # Source reference: the original text (if stored) and the autoencoder's own free-run decode of
    # the true latent (t=0, the decode ceiling).
    print("\n=== source latents (t=0, autoencoder ceiling) ===")
    for i in range(take):
        if sources is not None and i < len(sources):
            print(f"[{i}] SOURCE : {sources[i]!r}")
        word_count = int(mask[i].sum().item())
        lengths = resolve_lengths(i, z_data[i], word_count)
        ref = decode_text(autoencoder, z_data[i : i + 1], mask[i], latent_stats, args, dtype, device, lengths)
        print(f"[{i}] CEILING: {ref!r}")

    path = getattr(model.config, "path", "linear")
    for t_start in t_values:
        noise_frac = float(path_coefficients(t_start, path)[1])  # sigma(t_start)
        z_gen = teacher_forced_sample(model, z_data, mask, t_start, steps, device, dtype)
        mse, cos = recon_metrics(z_gen, z_data, mask)
        print(
            f"\n=== t_start={t_start:.2f}  noise_frac={noise_frac:.3f}  "
            f"recon_mse={mse:.4f}  cos_sim={cos:.3f} ==="
        )
        for i in range(take):
            word_count = int(mask[i].sum().item())
            lengths = resolve_lengths(i, z_gen[i], word_count)
            text = decode_text(autoencoder, z_gen[i : i + 1], mask[i], latent_stats, args, dtype, device, lengths)
            print(f"[{i}] {text!r}")


if __name__ == "__main__":
    main()
