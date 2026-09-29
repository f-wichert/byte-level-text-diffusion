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
    parser.add_argument(
        "--anchor-weight",
        default="0,1.0",
        help="Comma-separated weights for the decoder-CE term (0 = pure-MSE control).",
    )
    parser.add_argument(
        "--x0-clamp",
        type=float,
        default=8.0,
        help="Clamp x̂₀ to [-c, c] (standardized units) before decoding, for high-noise stability. 0 disables.",
    )
    parser.add_argument("--t-power", type=float, default=1.0, help="t=u**p (1=uniform; >1 oversamples high noise).")
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
        help="If set, save each weight's final model + embedded target to <save-path>_w<w>.pt.",
    )
    parser.add_argument(
        "--disable-self-cond",
        action="store_true",
        help="Build the DiT without self-conditioning (default-recommended: isolates the anchor lever).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def anchor_ce(
    autoencoder,
    x0_hat_std: torch.Tensor,
    byte_ids: torch.Tensor,
    boundary: torch.Tensor,
    labels: torch.Tensor,
    latent_stats: dict[str, torch.Tensor] | None,
    x0_clamp: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Cross-entropy of the real bytes decoded from x̂₀ through the frozen byte decoder.

    ``x0_hat_std`` is [1, L, D] in the model's standardized space; we clamp it (high-noise
    stability), de-standardize to the decoder's native latent space, and teacher-force the decoder
    on the real ``byte_ids`` with shifted ``labels``. Gradients flow to x̂₀ (decoder params frozen).
    """
    if x0_clamp and x0_clamp > 0:
        x0_hat_std = x0_hat_std.clamp(-x0_clamp, x0_clamp)
    z_native = x0_hat_std.float()
    if latent_stats is not None:
        z_native = z_native * latent_stats["std"] + latent_stats["mean"]
    z_native = z_native.to(dtype=dtype)
    out = autoencoder.forward_from_latents(
        byte_ids=byte_ids, z_words=[z_native], word_boundaries=[boundary], labels=labels,
    )
    return out["loss"]


def anchor_flow_loss(
    model,
    autoencoder,
    z_data: torch.Tensor,
    mask: torch.Tensor,
    byte_ids: torch.Tensor,
    boundary: torch.Tensor,
    labels: torch.Tensor,
    latent_stats: dict[str, torch.Tensor] | None,
    anchor_weight: float,
    x0_clamp: float,
    t_power: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float, float]:
    """masked-MSE (+ optional decoder-CE on x̂₀). Returns (total, mse, ce) with mse/ce as floats."""
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
    mse = masked_mse(prediction, target, mask)

    if anchor_weight <= 0:
        return mse, float(mse.item()), 0.0
    x0_hat = prediction if prediction_mode == "x0" else velocity_to_x0(z_t, prediction, t, path)
    ce = anchor_ce(autoencoder, x0_hat, byte_ids, boundary, labels, latent_stats, x0_clamp, dtype)
    total = mse + anchor_weight * ce
    return total, float(mse.item()), float(ce.item())


def main() -> None:
    args = parse_args()
    config = load_diffusion_config(args.config)
    seed_everything(args.seed)
    device = torch.device(args.device or config.training.device)
    dtype = _dtype_from_name(config.training.dtype)
    L = args.num_words
    weights = [float(v) for v in args.anchor_weight.split(",") if v.strip()]

    autoencoder = load_frozen_autoencoder(config, device)
    latent_stats = None
    if config.autoencoder.latent_stats_path:
        latent_stats = load_latent_stats(config.autoencoder.latent_stats_path, device, config.diffusion.latent_dim)

    # One target sequence (>= L words), standardized; plus the raw bytes/boundaries/labels for the CE.
    batches = cycle_batches(build_training_loader(config)[0])
    target = None
    source = ""
    word_lengths: list[int] = []
    byte_ids = boundary = labels = None
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
            byte_cut = int(bound[L].item())
            source = bytes(batch["byte_ids"][i, :byte_cut].tolist()).decode("utf-8", errors="replace")
            target = lb["z"][i, :L].clone().unsqueeze(0)  # [1, L, D] standardized
            # Raw decoder inputs for the L-word prefix; labels = next-byte, -100 past the cut (cf. data.py).
            byte_ids = batch["byte_ids"][i, :byte_cut].to(torch.long).unsqueeze(0)  # [1, byte_cut]
            boundary = bound[: L + 1].contiguous()  # [L+1], cumulative, ends at byte_cut
            labels = byte_ids.clone()
            labels[:, :-1] = byte_ids[:, 1:]
            labels[:, -1] = -100
            break

    mask = torch.ones((1, L), device=device, dtype=torch.bool)
    steps_sample = args.sampling_steps or config.diffusion.sampling_steps
    diffusion_cfg = dataclasses.replace(config.diffusion, self_conditioning=False) if args.disable_self_cond else config.diffusion
    print(f"overfit {L} words, {args.steps} steps; anchor weights {weights}; "
          f"x0_clamp={args.x0_clamp}; t_power={args.t_power}; self_cond={diffusion_cfg.self_conditioning}")
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
    # CE floor: decoding the TRUE latent -- the best CE the anchor term could reach.
    with torch.no_grad():
        ce_floor = anchor_ce(autoencoder, target, byte_ids, boundary, labels, latent_stats, args.x0_clamp, dtype)
    print(f"CEILING: {ceiling!r}  (best-possible decode of the true latent; judge SAMPLE against this)")
    print(f"CE floor (true latent): {float(ce_floor.item()):.4f}  (the anchor CE's achievable best)\n", flush=True)

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

    for w in weights:
        print(f"\n########## anchor-weight={w:g} ##########", flush=True)
        model = build_diffusion_backbone(dataclasses.replace(diffusion_cfg)).to(device=device, dtype=dtype)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=config.training.betas)
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, mse_v, ce_v = anchor_flow_loss(
                model, autoencoder, target, mask, byte_ids, boundary, labels,
                latent_stats, w, args.x0_clamp, args.t_power, dtype,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            optimizer.step()
            if args.probe_every and step % args.probe_every == 0:
                print(f"  step {step:>6} | loss {float(loss.item()):.4f} (mse {mse_v:.4f}  ce {ce_v:.4f})", flush=True)
                probe_and_print(model, f"step {step}")
        probe_and_print(model, "final")
        if args.save_path:
            out = f"{args.save_path}_w{w:g}.pt"
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
                    "anchor_weight": w,
                },
            )
            print(f"  saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
