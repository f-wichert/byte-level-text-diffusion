import argparse
import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_x0_by_t import x0_error_at_t
from scripts.sample_diffusion import decode_latents, sample_latents
from src.diffusion_model import build_diffusion_backbone
from src.diffusion_train import (
    encode_latent_batch,
    load_frozen_autoencoder,
    load_latent_stats,
    masked_mse,
    path_coefficients,
    velocity_to_x0,
)
from src.train import _dtype_from_name, build_training_loader, cycle_batches, move_batch_to_device
from src.utils import load_diffusion_config, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--t-power", default="1,6", help="Comma-separated p for t=u**p (1=uniform).")
    parser.add_argument("--num-words", type=int, default=24)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--bytes-per-word", type=int, default=6)
    parser.add_argument("--seed-byte", type=int, default=32)
    parser.add_argument("--t-values", default="0.02,0.05,0.10,0.15,0.20,0.30,0.50,0.70,0.90,0.98")
    parser.add_argument("--noise-draws", type=int, default=16)
    parser.add_argument(
        "--probe-every",
        type=int,
        default=0,
        help="If >0, run the per-t probe + decode a SAMPLE every N steps (trajectory vs steps).",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="If set, save each power's final model + embedded target to <save-path>_p<p>.pt "
        "(reloadable by diagnose_x0_by_t.py / compare_overfit_latents.py).",
    )
    parser.add_argument(
        "--disable-self-cond",
        action="store_true",
        help="Build the DiT without self-conditioning, to isolate the weighting lever from the "
        "self-cond feedback blow-up that NaNs under heavy high-noise emphasis.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def biased_flow_loss(model, z_data, mask, t_power: float) -> torch.Tensor:
    """flow_matching_loss with t drawn as u**t_power (oversamples high noise for t_power>1)."""
    noise = torch.randn_like(z_data)
    t = torch.rand(z_data.size(0), device=z_data.device) ** t_power
    path = getattr(model.config, "path", "linear")
    alpha, sigma, alpha_dot, sigma_dot = (c[:, None, None].to(z_data.dtype) for c in path_coefficients(t, path))
    z_t = alpha * z_data + sigma * noise
    prediction_mode = getattr(model.config, "prediction", "velocity")
    target = z_data if prediction_mode == "x0" else alpha_dot * z_data + sigma_dot * noise

    z_self = None
    if getattr(model.config, "self_conditioning", False) and torch.rand(()) < 0.5:
        with torch.no_grad():
            out0 = model(z_t, t, mask)
            x0_self = out0 if prediction_mode == "x0" else velocity_to_x0(z_t, out0, t, path)
            z_self = x0_self.to(z_data.dtype)
    prediction = model(z_t, t, mask, z_self)
    return masked_mse(prediction, target, mask)


def main() -> None:
    args = parse_args()
    config = load_diffusion_config(args.config)
    seed_everything(args.seed)
    device = torch.device(args.device or config.training.device)
    dtype = _dtype_from_name(config.training.dtype)
    L = args.num_words
    powers = [float(v) for v in args.t_power.split(",") if v.strip()]

    autoencoder = load_frozen_autoencoder(config, device)
    latent_stats = None
    if config.autoencoder.latent_stats_path:
        latent_stats = load_latent_stats(config.autoencoder.latent_stats_path, device, config.diffusion.latent_dim)

    # One target sequence (>= L words), standardized.
    batches = cycle_batches(build_training_loader(config)[0])
    target = None
    source = ""
    word_lengths: list[int] = []
    while target is None:
        batch = move_batch_to_device(next(batches), device)
        lb = encode_latent_batch(
            autoencoder, batch, max_words=config.diffusion.max_words,
            latent_dim=config.diffusion.latent_dim, dtype=dtype, device=device, latent_stats=latent_stats,
        )
        for i in range(lb["z"].size(0)):
            if int(lb["lengths"][i].item()) < L:
                continue
            bound = batch["word_boundaries"][i].to(torch.long)
            word_lengths = (bound[1 : L + 1] - bound[:L]).tolist()
            source = bytes(batch["byte_ids"][i, : int(bound[L].item())].tolist()).decode("utf-8", errors="replace")
            target = lb["z"][i, :L].clone().unsqueeze(0)  # [1, L, D]
            break

    mask = torch.ones((1, L), device=device, dtype=torch.bool)
    steps_sample = args.sampling_steps or config.diffusion.sampling_steps
    diffusion_cfg = dataclasses.replace(config.diffusion, self_conditioning=False) if args.disable_self_cond else config.diffusion
    print(f"overfit {L} words, {args.steps} steps; t-powers {powers}; self_cond={diffusion_cfg.self_conditioning}")
    print(f"SOURCE : {source!r}\n")

    def unstandardize(z_std_4096: torch.Tensor) -> torch.Tensor:
        z = z_std_4096.float().unsqueeze(0)
        if latent_stats is not None:
            z = z * latent_stats["std"] + latent_stats["mean"]
        return z.to(dtype=dtype)

    ceiling = decode_latents(
        autoencoder, unstandardize(target[0]), num_bytes=sum(word_lengths),
        bytes_per_word=args.bytes_per_word, seed_byte=args.seed_byte, device=device, word_lengths=word_lengths,
    )
    print(f"CEILING: {ceiling!r}  (best-possible decode of the true latent; judge SAMPLE against this)\n", flush=True)

    t_values = [float(v) for v in args.t_values.split(",") if v.strip()]
    self_cond = diffusion_cfg.self_conditioning

    def probe_and_print(model, label: str) -> None:
        """Per-t x̂₀ cosine (one compact line) + a decoded SAMPLE from noise, at a training checkpoint."""
        was_training = model.training
        model.eval()
        gen = torch.Generator(device=device).manual_seed(args.seed)
        cells = []
        for t_value in t_values:
            m = x0_error_at_t(model, target, mask, t_value, args.noise_draws, gen, device, dtype)
            cells.append(f"{t_value:.2f}={(m['cos_p2'] if self_cond else m['cos_p1']):.3f}")
        z_sample = sample_latents(model, 1, L, config.diffusion.latent_dim, steps_sample, device, dtype)
        sample = decode_latents(
            autoencoder, unstandardize(z_sample[0]), num_bytes=L * args.bytes_per_word,
            bytes_per_word=args.bytes_per_word, seed_byte=args.seed_byte, device=device,
        )
        print(f"  [{label}] cos x0(t): {'  '.join(cells)}")
        print(f"  [{label}] SAMPLE: {sample!r}", flush=True)
        if was_training:
            model.train()

    for p in powers:
        frac_high = float((torch.rand(100000) ** p < 0.05).float().mean())  # fraction of t<0.05
        print(f"\n########## t-power={p:g}  (P[t<0.05]={frac_high:.0%}) ##########", flush=True)
        model = build_diffusion_backbone(dataclasses.replace(diffusion_cfg)).to(device=device, dtype=dtype)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=config.training.betas)
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = biased_flow_loss(model, target, mask, p)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            optimizer.step()
            if args.probe_every and step % args.probe_every == 0:
                print(f"  step {step:>6} | loss {float(loss.item()):.4f}", flush=True)
                probe_and_print(model, f"step {step}")
        probe_and_print(model, "final")
        if args.save_path:
            out = f"{args.save_path}_p{p:g}.pt"
            save_checkpoint(
                out, model=model, step=args.steps,
                extra={
                    "config": dataclasses.asdict(dataclasses.replace(config, diffusion=diffusion_cfg)),
                    "latent_stats": (
                        {"mean": latent_stats["mean"].cpu(), "std": latent_stats["std"].cpu(),
                         "path": config.autoencoder.latent_stats_path}
                        if latent_stats is not None else None
                    ),
                    "overfit": {"z_data": target.cpu(), "mask": mask.cpu(), "sources": [source],
                                "word_lengths": [word_lengths], "num_words": L},
                    "t_power": p,
                },
            )
            print(f"  saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
