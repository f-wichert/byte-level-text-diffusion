import argparse
import math
import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sample_diffusion import decode_latents, sample_latents
from src.diffusion_model import build_diffusion_backbone
from src.diffusion_train import (
    encode_latent_batch,
    flow_matching_loss,
    load_frozen_autoencoder,
    load_latent_stats,
)
from src.train import (
    apply_tf32,
    build_training_loader,
    cycle_batches,
    diffusion_dtype_from_config,
    move_batch_to_device,
)
from src.utils import load_diffusion_config, module_dtype, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--num-samples", type=int, default=4, help="Sequences to overfit on.")
    parser.add_argument("--num-words", type=int, default=24, help="Words per sequence (common length).")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                        help="cosine = 5%% warmup then cosine decay to 2%% of peak (the combined-probe recipe).")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--bytes-per-word", type=int, default=6)
    parser.add_argument("--seed-byte", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-path",
        default=None,
        help="If set, save the overfit model + its exact target latents here "
        "(reloadable by scripts/compare_overfit_latents.py). Default: write nothing.",
    )
    parser.add_argument(
        "--oracle-prefix",
        type=int,
        default=None,
        help="If set, also sample with the true first N latents of each memorized sequence "
        "held clean (conditioning='clean') and report whether the continuation locks onto "
        "the right sequence (nearest memorized sequence by continuation-latent MSE).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_diffusion_config(args.config)
    seed_everything(args.seed)
    device = torch.device(args.device or config.training.device)
    apply_tf32(config.training)
    dtype = diffusion_dtype_from_config(config.training)
    L = args.num_words

    autoencoder = load_frozen_autoencoder(config, device)
    latent_stats = None
    if config.autoencoder.latent_stats_path:
        latent_stats = load_latent_stats(config.autoencoder.latent_stats_path, device, config.diffusion.latent_dim)

    # Collect num_samples real sequences that are at least L words long, truncated to exactly L.
    train_loader, _ = build_training_loader(config)
    batches = cycle_batches(train_loader)
    z_rows: list[torch.Tensor] = []      # [L, D] standardized target latents
    sources: list[str] = []              # original text, first L words
    word_lengths: list[list[int]] = []   # per-word byte lengths, for the ceiling decode
    byte_rows: list[list[int]] = []      # raw byte ids of the first L words (exact, pre-decode)
    while len(z_rows) < args.num_samples:
        batch = move_batch_to_device(next(batches), device)
        latent_batch = encode_latent_batch(
            autoencoder, batch,
            max_words=config.diffusion.max_words,
            latent_dim=config.diffusion.latent_dim,
            dtype=dtype, device=device, latent_stats=latent_stats,
        )
        boundaries = batch["word_boundaries"]
        for i in range(latent_batch["z"].size(0)):
            if int(latent_batch["lengths"][i].item()) < L:
                continue
            bound = boundaries[i].to(torch.long)
            lengths = (bound[1 : L + 1] - bound[:L]).tolist()
            byte_cut = int(bound[L].item())
            byte_rows.append(batch["byte_ids"][i, :byte_cut].tolist())
            sources.append(bytes(byte_rows[-1]).decode("utf-8", errors="replace"))
            word_lengths.append(lengths)
            z_rows.append(latent_batch["z"][i, :L].clone())
            if len(z_rows) >= args.num_samples:
                break

    z_data = torch.stack(z_rows, dim=0)                                  # [N, L, D]
    mask = torch.ones((z_data.size(0), L), device=device, dtype=torch.bool)
    print(f"overfitting on {z_data.size(0)} sequences x {L} words "
          f"(path={config.diffusion.path}, self_cond={getattr(config.diffusion, 'self_conditioning', False)})")

    # Fresh backbone, AdamW on the single fixed batch. --lr-schedule cosine mirrors
    # overfit_combined_probe (5% warmup -> cosine decay to 2% of peak); the passing
    # 06-30/WP1 gates all trained with that decay, constant LR never memorizes t->0.
    model = build_diffusion_backbone(config.diffusion).to(device=device, dtype=dtype)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=config.training.betas)

    def lr_at(step: int) -> float:
        if args.lr_schedule == "constant":
            return args.lr
        warmup = max(1, int(0.05 * args.steps))
        if step <= warmup:
            return args.lr * step / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        floor = 0.02 * args.lr
        return floor + (args.lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    for step in range(1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = flow_matching_loss(model, z_data, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            print(f"  step {step:>5} | loss {loss.item():.5f} | lr {lr_at(step):.2e}", flush=True)

    if args.save_path:
        save_checkpoint(
            args.save_path,
            model=model,
            step=args.steps,
            extra={
                "config": dataclasses.asdict(config),
                "latent_stats": (
                    {
                        "mean": latent_stats["mean"].cpu(),
                        "std": latent_stats["std"].cpu(),
                        "path": config.autoencoder.latent_stats_path,
                    }
                    if latent_stats is not None
                    else None
                ),
                "overfit": {
                    "z_data": z_data.cpu(),
                    "mask": mask.cpu(),
                    "sources": sources,
                    "byte_ids": byte_rows,
                    "word_lengths": word_lengths,
                    "num_words": L,
                },
            },
        )
        print(f"saved overfit model + targets to {args.save_path}", flush=True)

    # Decode helpers reverse the standardization the targets were stored in.
    def unstandardize(z_std: torch.Tensor) -> torch.Tensor:
        z = z_std.unsqueeze(0).float()
        if latent_stats is not None:
            z = z * latent_stats["std"] + latent_stats["mean"]
        # The decoder's dtype, not the flow's: under training.diffusion_dtype the two differ.
        return z.to(dtype=module_dtype(autoencoder, dtype))

    model.eval()
    steps = args.sampling_steps or config.diffusion.sampling_steps
    samples = sample_latents(
        model, num_samples=z_data.size(0), num_words=L,
        latent_dim=config.diffusion.latent_dim, steps=steps, device=device, dtype=dtype,
    )

    # Pure-noise memorization check in latent space (the WP1/06-30 gate criterion:
    # cos(z_gen, z*) flat >= ~0.95; decoded text below is only a secondary read --
    # the fixed byte grid missegments even ground-truth latents).
    print(f"\nsampling_steps={steps} | bytes-per-word grid={args.bytes_per_word} for SAMPLE decode")
    print("pure-noise memorization, cos(sample, target) per sequence:")
    for i in range(z_data.size(0)):
        cos_per_word = torch.nn.functional.cosine_similarity(
            samples[i].float(), z_data[i].float(), dim=-1
        )
        print(f"  seq {i + 1}: mean {cos_per_word.mean().item():.3f} | "
              f"min {cos_per_word.min().item():.3f} | "
              f"frac>=0.95 {(cos_per_word >= 0.95).float().mean().item():.2f}")
    print("note: the first 1-2 words are unreliable (decode is seeded with a space byte)")
    for i in range(z_data.size(0)):
        ceiling = decode_latents(
            autoencoder, unstandardize(z_data[i]),
            num_bytes=sum(word_lengths[i]), bytes_per_word=args.bytes_per_word,
            seed_byte=args.seed_byte, device=device, word_lengths=word_lengths[i],
        )
        sample = decode_latents(
            autoencoder, unstandardize(samples[i]),
            num_bytes=L * args.bytes_per_word, bytes_per_word=args.bytes_per_word,
            seed_byte=args.seed_byte, device=device,
        )
        print(f"\n===== sequence {i + 1} =====")
        print(f"SOURCE : {sources[i]!r}")
        print(f"CEILING: {ceiling!r}")
        print(f"SAMPLE : {sample!r}")

    # Oracle prefix probe: clamp the true first N latents clean and check the continuation
    # locks onto the right memorized sequence — the transport-of-information test that
    # unconditional-trained checkpoints fail (see notes/2026-07-04-*.md, diagnostic 2).
    if args.oracle_prefix:
        from src.sampling import sample_latents_conditional

        P = args.oracle_prefix
        if not 0 < P < L:
            raise SystemExit(f"--oracle-prefix must be in (0, {L}); got {P}.")
        z_known = torch.zeros_like(z_data)
        z_known[:, :P] = z_data[:, :P]
        known_mask = torch.zeros((z_data.size(0), L), dtype=torch.bool)
        known_mask[:, :P] = True
        prefix_samples = sample_latents_conditional(
            model, z_known, known_mask, steps, device, dtype, conditioning="clean"
        )
        print(f"\n===== oracle prefix probe (first {P} true latents clean; prefix bytes teacher-forced) =====")
        correct = 0
        for i in range(z_data.size(0)):
            cont = prefix_samples[i, P:].float()
            dists = [(cont - z_data[j, P:].float()).pow(2).mean().item() for j in range(z_data.size(0))]
            nearest = int(torch.tensor(dists).argmin().item())
            correct += int(nearest == i)
            # The prefix TEXT is known (its latents are clamped to truth), so teacher-force
            # its bytes: kills the seed-byte artifact and free-runs only the continuation,
            # matching prompt-mode generation. The latent-MSE lock-on above is decode-free.
            prefix_bytes = byte_rows[i][: sum(word_lengths[i][:P])]
            decoded = decode_latents(
                autoencoder, unstandardize(prefix_samples[i]),
                num_bytes=sum(word_lengths[i]), bytes_per_word=args.bytes_per_word,
                seed_byte=args.seed_byte, device=device, word_lengths=word_lengths[i],
                prompt_byte_ids=prefix_bytes,
            )
            print(f"\n----- sequence {i + 1} | nearest by continuation MSE: {nearest + 1} "
                  f"{'OK' if nearest == i else 'MISS'} | dists={[f'{d:.3f}' for d in dists]}")
            print(f"SOURCE       : {sources[i]!r}")
            print(f"PREFIX-SAMPLE: {decoded!r}")
        print(f"\noracle prefix result: {correct}/{z_data.size(0)} continuations locked onto the "
              f"correct sequence (pass criterion: >= {max(1, z_data.size(0) - 1)})")


if __name__ == "__main__":
    main()
