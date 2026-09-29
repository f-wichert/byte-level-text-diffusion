import argparse
import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_x0_by_t import x0_error_at_t
from scripts.sample_diffusion import _clip_boundaries, decode_latents, sample_latents
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
    parser.add_argument("--anchor-weight", default="0.3,1.0", help="Comma-separated CE weights.")
    parser.add_argument("--x0-clamp", type=float, default=8.0, help="Clamp x̂₀ to [-c,c] (std units) before decode. 0 disables.")
    parser.add_argument("--tf-start", type=float, default=1.0, help="Teacher-forcing ratio at step 0 (1=Stage-2 sanity).")
    parser.add_argument("--tf-end", type=float, default=0.0, help="Teacher-forcing ratio at the end (0=pure free-run).")
    parser.add_argument("--tf-anneal-frac", type=float, default=0.7, help="Fraction of steps over which tf anneals start->end.")
    parser.add_argument("--num-words", type=int, default=12)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--bytes-per-word", type=int, default=6)
    parser.add_argument("--seed-byte", type=int, default=32)
    parser.add_argument("--t-values", default="0.02,0.05,0.10,0.15,0.20,0.30,0.50,0.70,0.90,0.98")
    parser.add_argument("--noise-draws", type=int, default=16)
    parser.add_argument("--probe-every", type=int, default=0, help="If >0, per-t probe + SAMPLE every N steps.")
    parser.add_argument("--save-path", default=None, help="If set, save each weight to <save-path>_w<w>.pt.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def free_run_context(
    autoencoder,
    z_native: torch.Tensor,
    real_bytes: list[int],
    full_boundaries: torch.Tensor,
    num_bytes: int,
    seed_byte: int,
    tf_ratio: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Autoregressive byte context from x̂₀ (native latents), detached. Each position is the real byte
    with prob ``tf_ratio`` else the model's own argmax -> a self-generated context for the CE pass."""
    generated = [int(seed_byte) % 256]
    while len(generated) < num_bytes:
        cur = len(generated)
        ids = torch.tensor(generated, device=device, dtype=torch.long).unsqueeze(0)
        boundaries = _clip_boundaries(full_boundaries, cur)
        word_count = min(boundaries.numel() - 1, z_native.size(1))
        boundaries = boundaries[: word_count + 1]
        logits = autoencoder.decode(ids, [z_native[:, :word_count, :]], [boundaries])
        nxt_pred = int(logits[0, cur - 1].argmax(dim=-1).item())
        if tf_ratio > 0 and float(torch.rand((), generator=generator, device=device).item()) < tf_ratio:
            generated.append(int(real_bytes[cur]))
        else:
            generated.append(nxt_pred)
    return torch.tensor(generated[:num_bytes], device=device, dtype=torch.long).unsqueeze(0)


def freeanchor_loss(
    model,
    autoencoder,
    z_data: torch.Tensor,
    mask: torch.Tensor,
    real_bytes: list[int],
    real_labels: torch.Tensor,
    full_boundaries: torch.Tensor,
    boundary: torch.Tensor,
    num_bytes: int,
    latent_stats: dict[str, torch.Tensor] | None,
    anchor_weight: float,
    x0_clamp: float,
    tf_ratio: float,
    seed_byte: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float, float]:
    """masked-MSE + w * free-running decoder-CE (context self-generated, scored vs real bytes)."""
    noise = torch.randn_like(z_data)
    t = torch.rand(z_data.size(0), device=z_data.device)
    path = getattr(model.config, "path", "linear")
    alpha, sigma, alpha_dot, sigma_dot = (c[:, None, None].to(z_data.dtype) for c in path_coefficients(t, path))
    z_t = alpha * z_data + sigma * noise
    prediction_mode = getattr(model.config, "prediction", "velocity")
    target = z_data if prediction_mode == "x0" else alpha_dot * z_data + sigma_dot * noise
    prediction = model(z_t, t, mask)
    mse = masked_mse(prediction, target, mask)
    if anchor_weight <= 0:
        return mse, float(mse.item()), 0.0

    x0_hat = prediction if prediction_mode == "x0" else velocity_to_x0(z_t, prediction, t, path)
    if x0_clamp and x0_clamp > 0:
        x0_hat = x0_hat.clamp(-x0_clamp, x0_clamp)
    z_native = x0_hat.float()
    if latent_stats is not None:
        z_native = z_native * latent_stats["std"] + latent_stats["mean"]
    z_native = z_native.to(dtype=dtype)

    context = free_run_context(
        autoencoder, z_native.detach(), real_bytes, full_boundaries, num_bytes,
        seed_byte, tf_ratio, generator, device,
    )
    out = autoencoder.forward_from_latents(
        byte_ids=context, z_words=[z_native], word_boundaries=[boundary], labels=real_labels,
    )
    ce = out["loss"]
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

    batches = cycle_batches(build_training_loader(config)[0])
    target = None
    source = ""
    word_lengths: list[int] = []
    real_bytes: list[int] = []
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
            real_bytes = batch["byte_ids"][i, :byte_cut].to(torch.long).tolist()
            source = bytes(real_bytes).decode("utf-8", errors="replace")
            target = lb["z"][i, :L].clone().unsqueeze(0)  # [1, L, D] standardized
            boundary = bound[: L + 1].contiguous()         # [L+1] cumulative, ends at byte_cut
            break

    num_bytes = len(real_bytes)
    full_boundaries = boundary.to(torch.int32)
    real_ids = torch.tensor(real_bytes, device=device, dtype=torch.long).unsqueeze(0)  # [1, T]
    real_labels = real_ids.clone()
    real_labels[:, :-1] = real_ids[:, 1:]
    real_labels[:, -1] = -100  # next-byte targets (cf. data.py); -100 past the cut
    mask = torch.ones((1, L), device=device, dtype=torch.bool)
    steps_sample = args.sampling_steps or config.diffusion.sampling_steps
    diffusion_cfg = dataclasses.replace(config.diffusion, self_conditioning=False)
    print(f"overfit {L} words / {num_bytes} bytes, {args.steps} steps; anchor weights {weights}; "
          f"x0_clamp={args.x0_clamp}; tf {args.tf_start}->{args.tf_end} over {args.tf_anneal_frac:.0%}; self_cond=False")
    print(f"SOURCE : {source!r}\n")

    def unstandardize(z_std: torch.Tensor) -> torch.Tensor:
        z = z_std.float().unsqueeze(0)
        if latent_stats is not None:
            z = z * latent_stats["std"] + latent_stats["mean"]
        return z.to(dtype=dtype)

    def to_native(z_std_LD: torch.Tensor) -> torch.Tensor:
        z = z_std_LD.float()
        if latent_stats is not None:
            z = z * latent_stats["std"] + latent_stats["mean"]
        return z.to(dtype=dtype)

    ceiling = decode_latents(
        autoencoder, unstandardize(target[0]), num_bytes=num_bytes,
        bytes_per_word=args.bytes_per_word, seed_byte=args.seed_byte, device=device, word_lengths=word_lengths,
    )
    with torch.no_grad():
        ce_floor = autoencoder.forward_from_latents(
            byte_ids=real_ids, z_words=[to_native(target[0]).unsqueeze(0)], word_boundaries=[boundary], labels=real_labels,
        )["loss"]
    print(f"CEILING: {ceiling!r}")
    print(f"CE floor (true latent, teacher-forced): {float(ce_floor.item()):.4f}\n", flush=True)

    t_values = [float(v) for v in args.t_values.split(",") if v.strip()]

    def probe_and_print(model, label: str) -> None:
        was_training = model.training
        model.eval()
        gen = torch.Generator(device=device).manual_seed(args.seed)
        cells = []
        for t_value in t_values:
            m = x0_error_at_t(model, target, mask, t_value, args.noise_draws, gen, device, dtype)
            cells.append(f"{t_value:.2f}={m['cos_p1']:.3f}")
        z_sample = sample_latents(model, 1, L, config.diffusion.latent_dim, steps_sample, device, dtype)
        sample = decode_latents(
            autoencoder, unstandardize(z_sample[0]), num_bytes=num_bytes,
            bytes_per_word=args.bytes_per_word, seed_byte=args.seed_byte, device=device, word_lengths=word_lengths,
        )
        print(f"  [{label}] cos x0(t): {'  '.join(cells)}")
        print(f"  [{label}] SAMPLE: {sample!r}", flush=True)
        if was_training:
            model.train()

    anneal_steps = max(1, int(args.tf_anneal_frac * args.steps))
    for w in weights:
        print(f"\n########## anchor-weight={w:g} ##########", flush=True)
        model = build_diffusion_backbone(dataclasses.replace(diffusion_cfg)).to(device=device, dtype=dtype)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=config.training.betas)
        roll_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
        for step in range(1, args.steps + 1):
            frac = min(1.0, step / anneal_steps)
            tf_ratio = args.tf_start + (args.tf_end - args.tf_start) * frac
            optimizer.zero_grad(set_to_none=True)
            loss, mse_v, ce_v = freeanchor_loss(
                model, autoencoder, target, mask, real_bytes, real_labels, full_boundaries, boundary,
                num_bytes, latent_stats, w, args.x0_clamp, tf_ratio, args.seed_byte, roll_gen, device, dtype,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            optimizer.step()
            if args.probe_every and step % args.probe_every == 0:
                print(f"  step {step:>6} | tf {tf_ratio:.2f} | loss {float(loss.item()):.4f} (mse {mse_v:.4f}  ce {ce_v:.4f})", flush=True)
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
