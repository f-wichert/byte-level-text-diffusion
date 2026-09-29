import argparse
import dataclasses
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_teacher_forced import teacher_forced_sample
from scripts.diagnose_x0_by_t import x0_error_at_t
from scripts.overfit_anchor_probe import anchor_ce
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
from src.utils import create_logger, load_diffusion_config, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--num-words", type=int, default=24)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak LR (after warmup).")
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--min-lr-frac", type=float, default=0.02, help="Final LR as a fraction of peak.")
    parser.add_argument("--t-power", type=float, default=2.2, help="t=u**p; p=2.2 => ~25%% of mass at t<0.05.")
    parser.add_argument("--ce-weight", type=float, default=0.3, help="Decoder-CE anchor weight (0 disables).")
    parser.add_argument("--self-cond", action="store_true", help="Enable self-conditioning (fed-back x̂₀ clamped for stability).")
    parser.add_argument("--x0-clamp", type=float, default=8.0)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--bytes-per-word", type=int, default=6)
    parser.add_argument("--seed-byte", type=int, default=32)
    parser.add_argument("--t-values", default="0.02,0.05,0.10,0.15,0.20,0.30,0.50,0.70,0.90,0.98")
    parser.add_argument("--noise-draws", type=int, default=16)
    parser.add_argument("--probe-every", type=int, default=1000)
    parser.add_argument(
        "--sdedit-t",
        default="0.0,0.05,0.10,0.15,0.20,0.40,0.60,0.80",
        help="SDEdit start times for the final-model 'recover from noise level' decode: noise the "
        "learned latent to each t_start and decode (small t = high noise). Logged into the samples table.",
    )
    parser.add_argument("--log-every", type=int, default=25, help="Step interval for scalar loss logging to wandb.")
    parser.add_argument("--wandb", action="store_true", help="Log losses + per-t x̂₀ cosines + samples to wandb (uses config training.wandb_* settings).")
    parser.add_argument("--wandb-mode", default=None, help="Override config wandb_mode (online/offline/disabled).")
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def combined_loss(model, autoencoder, z_data, mask, byte_ids, boundary, labels,
                  latent_stats, ce_weight, x0_clamp, t_power, dtype):
    """masked-MSE with high-noise-biased t (+ optional self-cond and teacher-forced decoder-CE on x̂₀)."""
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
            if x0_clamp and x0_clamp > 0:  # clamp the fed-back estimate (prevents the high-noise NaN)
                x0_self = x0_self.clamp(-x0_clamp, x0_clamp)
            z_self = x0_self.to(z_data.dtype)
    prediction = model(z_t, t, mask, z_self)
    mse = masked_mse(prediction, target, mask)
    if ce_weight <= 0 or prediction_mode != "x0":
        return mse, float(mse.item()), 0.0
    ce = anchor_ce(autoencoder, prediction, byte_ids, boundary, labels, latent_stats, x0_clamp, dtype)
    return mse + ce_weight * ce, float(mse.item()), float(ce.item())


def main() -> None:
    args = parse_args()
    config = load_diffusion_config(args.config)
    seed_everything(args.seed)
    device = torch.device(args.device or config.training.device)
    dtype = _dtype_from_name(config.training.dtype)
    L = args.num_words

    autoencoder = load_frozen_autoencoder(config, device)
    latent_stats = None
    if config.autoencoder.latent_stats_path:
        latent_stats = load_latent_stats(config.autoencoder.latent_stats_path, device, config.diffusion.latent_dim)

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
            target = lb["z"][i, :L].clone().unsqueeze(0)
            byte_ids = batch["byte_ids"][i, :byte_cut].to(torch.long).unsqueeze(0)
            boundary = bound[: L + 1].contiguous()
            labels = byte_ids.clone()
            labels[:, :-1] = byte_ids[:, 1:]
            labels[:, -1] = -100
            break

    mask = torch.ones((1, L), device=device, dtype=torch.bool)
    steps_sample = args.sampling_steps or config.diffusion.sampling_steps
    diffusion_cfg = dataclasses.replace(config.diffusion, self_conditioning=args.self_cond)
    self_cond = diffusion_cfg.self_conditioning
    frac_high = float((torch.rand(200000) ** args.t_power < 0.05).float().mean())
    print(f"overfit {L} words, {args.steps} steps; LR warmup {args.warmup_frac:.0%}->peak {args.lr:g}->cosine "
          f"decay to {args.min_lr_frac:.0%}; t-power {args.t_power:g} (P[t<0.05]={frac_high:.0%}); "
          f"ce-weight {args.ce_weight:g}; self_cond={self_cond}")
    print(f"SOURCE : {source!r}\n")

    def unstandardize(z_std: torch.Tensor) -> torch.Tensor:
        z = z_std.float().unsqueeze(0)
        if latent_stats is not None:
            z = z * latent_stats["std"] + latent_stats["mean"]
        return z.to(dtype=dtype)

    ceiling = decode_latents(
        autoencoder, unstandardize(target[0]), num_bytes=sum(word_lengths),
        bytes_per_word=args.bytes_per_word, seed_byte=args.seed_byte, device=device, word_lengths=word_lengths,
    )
    print(f"CEILING: {ceiling!r}\n", flush=True)

    t_values = [float(v) for v in args.t_values.split(",") if v.strip()]

    logger = None
    if args.wandb:
        if args.wandb_mode:
            config.training.wandb_mode = args.wandb_mode
        run_config = dataclasses.asdict(dataclasses.replace(config, diffusion=diffusion_cfg))
        run_config["recipe"] = {
            "t_power": args.t_power, "ce_weight": args.ce_weight, "lr": args.lr,
            "warmup_frac": args.warmup_frac, "min_lr_frac": args.min_lr_frac,
            "self_cond": self_cond, "num_words": L, "steps": args.steps,
        }
        logger = create_logger(config.training, config.training.output_dir, run_config)
    sample_rows: list[tuple[str, str]] = []

    def probe_and_print(model, label: str, step: int | None = None) -> None:
        was_training = model.training
        model.eval()
        gen = torch.Generator(device=device).manual_seed(args.seed)
        cells, cos_log = [], {}
        for t_value in t_values:
            m = x0_error_at_t(model, target, mask, t_value, args.noise_draws, gen, device, dtype)
            cos = m["cos_p2"] if self_cond else m["cos_p1"]
            cells.append(f"{t_value:.2f}={cos:.3f}")
            cos_log[f"cos_x0/t{t_value:.2f}"] = cos
        z_sample = sample_latents(model, 1, L, config.diffusion.latent_dim, steps_sample, device, dtype)
        sample = decode_latents(
            autoencoder, unstandardize(z_sample[0]), num_bytes=L * args.bytes_per_word,
            bytes_per_word=args.bytes_per_word, seed_byte=args.seed_byte, device=device,
        )
        print(f"  [{label}] cos x0(t): {'  '.join(cells)}")
        print(f"  [{label}] SAMPLE: {sample!r}", flush=True)
        if logger is not None and step is not None:
            logger.log(cos_log, step)
        sample_rows.append((label, sample))
        if was_training:
            model.train()

    model = build_diffusion_backbone(dataclasses.replace(diffusion_cfg)).to(device=device, dtype=dtype)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=config.training.betas)
    warmup = max(1, int(args.warmup_frac * args.steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return args.min_lr_frac + (1.0 - args.min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_at)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, mse_v, ce_v = combined_loss(
            model, autoencoder, target, mask, byte_ids, boundary, labels,
            latent_stats, args.ce_weight, args.x0_clamp, args.t_power, dtype,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        if logger is not None and step % args.log_every == 0:
            logger.log({"train/loss": float(loss.item()), "train/mse": mse_v, "train/ce": ce_v,
                        "train/lr": optimizer.param_groups[0]["lr"]}, step)
        if args.probe_every and step % args.probe_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  step {step:>6} | lr {lr_now:.2e} | loss {float(loss.item()):.4f} (mse {mse_v:.4f}  ce {ce_v:.4f})", flush=True)
            probe_and_print(model, f"step {step}", step)
    probe_and_print(model, "final", args.steps)

    # SDEdit "recover from each noise level": noise the learned latent to each t_start and decode
    # (small t = high noise; t=0 is pure noise = the unconditional SAMPLE). Final model only.
    sdedit_rows: list[tuple[str, str]] = []
    sdedit_ts = [float(v) for v in args.sdedit_t.split(",") if v.strip()]
    if sdedit_ts:
        model.eval()
        path = getattr(model.config, "path", "linear")
        valid = mask.float().sum().clamp_min(1.0)
        print("\nSDEdit decode (noise the learned latent to t_start, integrate to 1):", flush=True)
        with torch.no_grad():
            for t_start in sdedit_ts:
                noise_frac = float(path_coefficients(t_start, path)[1])  # sigma(t_start)
                z_gen = teacher_forced_sample(model, target, mask, t_start, steps_sample, device, dtype)
                cos = float((torch.cosine_similarity(z_gen.float(), target.float(), dim=-1) * mask.float()).sum() / valid)
                text = decode_latents(
                    autoencoder, unstandardize(z_gen[0]), num_bytes=sum(word_lengths),
                    bytes_per_word=args.bytes_per_word, seed_byte=args.seed_byte, device=device,
                    word_lengths=word_lengths,
                )
                label = f"noise t={t_start:.2f} (frac {noise_frac:.3f}, cos {cos:.3f})"
                print(f"  [{label}] {text!r}", flush=True)
                sdedit_rows.append((label, text))

    if args.save_path:
        out = f"{args.save_path}.pt"
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
                "recipe": {"t_power": args.t_power, "ce_weight": args.ce_weight,
                           "lr": args.lr, "warmup_frac": args.warmup_frac, "min_lr_frac": args.min_lr_frac},
            },
        )
        print(f"  saved -> {out}", flush=True)

    if logger is not None:
        if hasattr(logger, "run"):  # WandbLogger: log the evolving samples + source/ceiling references
            import wandb

            table = wandb.Table(columns=["label", "sample"])
            for lbl, smp in sample_rows:
                table.add_data(lbl, smp)
            for lbl, smp in sdedit_rows:
                table.add_data(lbl, smp)
            logger.run.summary["source"] = source
            logger.run.summary["ceiling"] = ceiling
            logger.run.log({"samples": table})
        logger.close()


if __name__ == "__main__":
    main()
