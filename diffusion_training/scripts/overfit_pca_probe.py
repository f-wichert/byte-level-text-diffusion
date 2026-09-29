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
    flow_matching_loss,
    load_frozen_autoencoder,
    load_latent_stats,
)
from src.train import _dtype_from_name, build_training_loader, cycle_batches, move_batch_to_device
from src.utils import load_diffusion_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--k", default="256,128", help="Comma-separated PCA dims to test.")
    parser.add_argument("--num-words", type=int, default=24, help="Words in the overfit sequence.")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pca-fit-words", type=int, default=4000, help="Real word-latents to fit PCA on.")
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--bytes-per-word", type=int, default=6)
    parser.add_argument("--seed-byte", type=int, default=32)
    parser.add_argument("--t-values", default="0.02,0.05,0.10,0.15,0.20,0.30,0.50,0.70,0.90,0.98")
    parser.add_argument("--noise-draws", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def fit_pca(x: torch.Tensor, max_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-(max_k) PCA basis of standardized word-latents ``x`` [M, D] (float32).

    Returns the centering mean [D] and an orthonormal component matrix V [D, max_k]
    (columns ordered by decreasing variance). Whitening scales are derived per-k by the caller.
    """
    mean = x.mean(dim=0)
    q = min(max_k + 32, x.size(1), x.size(0))  # oversample for accuracy of the top components
    _, _, v = torch.pca_lowrank(x, q=q, center=True, niter=4)
    return mean, v[:, :max_k].contiguous()


def main() -> None:
    args = parse_args()
    config = load_diffusion_config(args.config)
    seed_everything(args.seed)
    device = torch.device(args.device or config.training.device)
    dtype = _dtype_from_name(config.training.dtype)
    L = args.num_words
    k_list = [int(v) for v in args.k.split(",") if v.strip()]
    max_k = max(k_list)

    autoencoder = load_frozen_autoencoder(config, device)
    latent_stats = None
    if config.autoencoder.latent_stats_path:
        latent_stats = load_latent_stats(config.autoencoder.latent_stats_path, device, config.diffusion.latent_dim)

    # Collect (a) a broad pool of standardized word-latents to fit PCA, and (b) one target sequence.
    train_loader, _ = build_training_loader(config)
    batches = cycle_batches(train_loader)
    pool: list[torch.Tensor] = []
    pool_words = 0
    target: torch.Tensor | None = None
    source = ""
    word_lengths: list[int] = []
    while pool_words < args.pca_fit_words or target is None:
        batch = move_batch_to_device(next(batches), device)
        lb = encode_latent_batch(
            autoencoder, batch, max_words=config.diffusion.max_words,
            latent_dim=config.diffusion.latent_dim, dtype=dtype, device=device, latent_stats=latent_stats,
        )
        for i in range(lb["z"].size(0)):
            n = int(lb["lengths"][i].item())
            if n <= 0:
                continue
            pool.append(lb["z"][i, :n].float())
            pool_words += n
            if target is None and n >= L:
                bound = batch["word_boundaries"][i].to(torch.long)
                word_lengths = (bound[1 : L + 1] - bound[:L]).tolist()
                byte_cut = int(bound[L].item())
                source = bytes(batch["byte_ids"][i, :byte_cut].tolist()).decode("utf-8", errors="replace")
                target = lb["z"][i, :L].clone()  # [L, D] standardized

    pool_t = torch.cat(pool, dim=0)[: args.pca_fit_words]  # [M, D] standardized, float32
    print(f"PCA fit on {pool_t.size(0)} word-latents; testing k in {k_list}; overfit {L} words, {args.steps} steps")
    print(f"SOURCE : {source!r}\n")

    mean, components = fit_pca(pool_t, max_k)            # mean [D], components [D, max_k]
    total_var = (pool_t - mean).pow(2).sum().item()
    mask = torch.ones((1, L), device=device, dtype=torch.bool)
    steps_sample = args.sampling_steps or config.diffusion.sampling_steps

    def unstandardize(z_std_4096: torch.Tensor) -> torch.Tensor:
        z = z_std_4096.float().unsqueeze(0)  # [L, D] -> [1, L, D] for decode_latents
        if latent_stats is not None:
            z = z * latent_stats["std"] + latent_stats["mean"]
        return z.to(dtype=dtype)

    for k in k_list:
        vk = components[:, :k]                            # [D, k]
        coords = (pool_t - mean) @ vk                     # [M, k]
        coord_std = coords.std(dim=0).clamp_min(1e-6)     # whitening scale [k]
        explained = coords.pow(2).sum().item() / total_var

        def project(z_std_4096: torch.Tensor) -> torch.Tensor:
            return (((z_std_4096.float() - mean) @ vk) / coord_std).to(dtype)

        def invert(z_white_k: torch.Tensor) -> torch.Tensor:
            return ((z_white_k.float() * coord_std) @ vk.t() + mean).to(dtype)

        z_k = project(target).unsqueeze(0)                # [1, L, k] whitened target

        # Fresh k-dim DiT, same diffusion settings (path/x0/self-cond), constant-LR overfit.
        cfg_k = dataclasses.replace(config.diffusion, latent_dim=k)
        model = build_diffusion_backbone(cfg_k).to(device=device, dtype=dtype)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=config.training.betas)
        last = 0.0
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = flow_matching_loss(model, z_k, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            optimizer.step()
            last = float(loss.item())
        model.eval()

        print(f"\n########## k={k}  (explained var {explained:.1%}, final loss {last:.4f}) ##########")

        # Per-t x0 recovery in k-dim (the Stage-0 curve).
        gen = torch.Generator(device=device).manual_seed(args.seed)
        self_cond = getattr(model.config, "self_conditioning", False)
        print(f"{'t':>6} {'cos_in':>8} {'cos x0':>8} {'relL2':>8}")
        for t_value in [float(v) for v in args.t_values.split(",") if v.strip()]:
            m = x0_error_at_t(model, z_k, mask, t_value, args.noise_draws, gen, device, dtype)
            cos = m["cos_p2"] if self_cond else m["cos_p1"]
            rl2 = m["rel_l2_p2"] if self_cond else m["rel_l2_p1"]
            print(f"{t_value:>6.2f} {m['cos_in']:>8.3f} {cos:>8.3f} {rl2:>8.3f}")

        # Decodes: full ceiling, PCA-truncated ceiling, and a sample from noise.
        ceiling_full = decode_latents(
            autoencoder, unstandardize(target), num_bytes=sum(word_lengths),
            bytes_per_word=args.bytes_per_word, seed_byte=args.seed_byte, device=device, word_lengths=word_lengths,
        )
        ceiling_pca = decode_latents(
            autoencoder, unstandardize(invert(z_k[0])), num_bytes=sum(word_lengths),
            bytes_per_word=args.bytes_per_word, seed_byte=args.seed_byte, device=device, word_lengths=word_lengths,
        )
        z_sample_k = sample_latents(model, 1, L, k, steps_sample, device, dtype)
        sample = decode_latents(
            autoencoder, unstandardize(invert(z_sample_k[0])), num_bytes=L * args.bytes_per_word,
            bytes_per_word=args.bytes_per_word, seed_byte=args.seed_byte, device=device,
        )
        print(f"CEILING_full: {ceiling_full!r}")
        print(f"CEILING_pca : {ceiling_pca!r}")
        print(f"SAMPLE      : {sample!r}")


if __name__ == "__main__":
    main()
